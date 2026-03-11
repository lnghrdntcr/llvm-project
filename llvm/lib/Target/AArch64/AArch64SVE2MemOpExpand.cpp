//===- AArch64SVE2MemOpExpand.cpp - SVE2 memset/memcpy expansion ----------===//
// Pre-regalloc expansion of SVE2MemorySet/Copy pseudos into real SVE2
// load/store loops using virtual registers.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

#define PASS_NAME "AArch64 SVE2 memset/memcpy pre-regalloc expansion"
#define DEBUG_TYPE "aarch64-sve2-memop-expand"

namespace {

class AArch64SVE2MemOpExpand : public MachineFunctionPass {
public:
  static char ID;
  AArch64SVE2MemOpExpand() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return PASS_NAME; }

private:
  bool expandSet(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandCopy(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
};

char AArch64SVE2MemOpExpand::ID = 0;

} // end anonymous namespace

bool AArch64SVE2MemOpExpand::runOnMachineFunction(MachineFunction &MF) {

  const AArch64Subtarget &ST = MF.getSubtarget<AArch64Subtarget>();
  if (!ST.hasSVE2())
    return false;

  bool Changed = false;

  // Collect pseudos first to avoid iterator invalidation.
  SmallVector<MachineInstr *, 4> ToExpand;
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      switch (MI.getOpcode()) {
      case AArch64::SVE2MemorySetPseudo:
      case AArch64::SVE2MemorySetNTPseudo:
      // case AArch64::SVE2MemoryCopyPseudo:
      // case AArch64::SVE2MemoryCopyNTPseudo:
      // case AArch64::SVE2MemoryMovePseudo:
      // case AArch64::SVE2MemoryMoveNTPseudo:
        ToExpand.push_back(&MI);
        break;
      default:
        break;
      }

  for (MachineInstr *MI : ToExpand) {
    MachineBasicBlock &MBB = *MI->getParent();
    switch (MI->getOpcode()) {
    case AArch64::SVE2MemorySetPseudo:
    case AArch64::SVE2MemorySetNTPseudo:
      Changed |= expandSet(MBB, MI);
      break;
    // case AArch64::SVE2MemoryCopyPseudo:
    // case AArch64::SVE2MemoryCopyNTPseudo:
    // case AArch64::SVE2MemoryMovePseudo:
    // case AArch64::SVE2MemoryMoveNTPseudo:
    //   Changed |= expandCopy(MBB, MI);
    //   break;
    default:
      llvm_unreachable("Unexpected opcode");
    }
  }

  return Changed;
}

bool AArch64SVE2MemOpExpand::expandSet(
    MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBI) {

  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();
  unsigned Opcode = MI.getOpcode();

  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  bool IsNT = (Opcode == AArch64::SVE2MemorySetNTPseudo);

  Register DstReg  = MI.getOperand(2).getReg();
  Register SizeReg = MI.getOperand(3).getReg();
  Register ValReg  = MI.getOperand(4).getReg();

  auto *GPR64   = &AArch64::GPR64RegClass;
  auto *GPR64sp = &AArch64::GPR64spRegClass;
  auto *GPR32sp = &AArch64::GPR32spRegClass;
  auto *ZRC     = &AArch64::ZPRRegClass;
  auto *PRC     = &AArch64::PPR_3bRegClass;

  auto New64   = [&]() { return MRI.createVirtualRegister(GPR64); };
  auto New64sp = [&]() { return MRI.createVirtualRegister(GPR64sp); };
  auto New32sp = [&]() { return MRI.createVirtualRegister(GPR32sp); };
  auto NewZ    = [&]() { return MRI.createVirtualRegister(ZRC); };
  auto NewP    = [&]() { return MRI.createVirtualRegister(PRC); };

  Register Pred   = NewP();
  Register Vec    = NewZ();
  // Dst0 must be GPR64sp to accept the incoming address register via COPY
  Register Dst0   = New64sp();
  Register Loop0  = New64();
  Register Loop1  = New64();
  Register Tail0  = New64();
  Register Tail1  = New64();
  Register VLen   = New64();
  // DUP_ZR_B requires GPR32sp
  Register Val32  = New32sp();

  constexpr int Unroll = 8;
  constexpr auto EffectiveVL = AArch64SVEPredPattern::vl8;

  MachineBasicBlock *InitBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *FastBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *TailBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *ExitBB = MF.CreateMachineBasicBlock(MBB.getBasicBlock());

  MF.insert(++MBB.getIterator(), InitBB);
  MF.insert(++InitBB->getIterator(), FastBB);
  MF.insert(++FastBB->getIterator(), TailBB);
  MF.insert(++TailBB->getIterator(), ExitBB);

  ExitBB->splice(ExitBB->end(), &MBB, std::next(MBBI), MBB.end());
  ExitBB->transferSuccessors(&MBB);

   // After transferSuccessors, any PHIs in ExitBB's successors that
   // referenced MBB as their incoming block now need to reference ExitBB,
   // since MBB's successors (and their PHI edges) moved to ExitBB.
   for (MachineBasicBlock *Succ : ExitBB->successors()) {
     for (MachineInstr &PHI : Succ->phis()) {
       for (unsigned i = 1; i < PHI.getNumOperands(); i += 2) {
         if (PHI.getOperand(i + 1).getMBB() == &MBB)
           PHI.getOperand(i + 1).setMBB(ExitBB);
       }
     }
   }

  MBB.addSuccessor(InitBB);
  InitBB->addSuccessor(FastBB);
  InitBB->addSuccessor(TailBB);
  FastBB->addSuccessor(FastBB);
  FastBB->addSuccessor(TailBB);
  TailBB->addSuccessor(TailBB);
  TailBB->addSuccessor(ExitBB);

  auto B = [&](MachineBasicBlock *BB, unsigned Opc) {
    return BuildMI(*BB, BB->end(), DL, TII->get(Opc));
  };

  /* -------- Init -------- */

  B(InitBB, TargetOpcode::COPY).addDef(Dst0).addUse(DstReg);

  B(InitBB, AArch64::PTRUE_B)
      .addDef(Pred)
      .addImm(AArch64SVEPredPattern::all);

  B(InitBB, TargetOpcode::COPY)
      .addDef(Val32)
      .addUse(ValReg, 0, AArch64::sub_32);

  // DUP_ZR_B expects GPR32sp
  B(InitBB, AArch64::DUP_ZR_B)
      .addDef(Vec)
      .addUse(Val32);

  B(InitBB, AArch64::CNTB_XPiI)
      .addDef(VLen)
      .addImm(AArch64SVEPredPattern::all)
      .addImm(1);

  B(InitBB, AArch64::UBFMXri)
      .addDef(Loop0)
      .addUse(SizeReg)
      .addImm(3)
      .addImm(63);

  B(InitBB, AArch64::UDIVXr)
      .addDef(Loop1)
      .addUse(Loop0)
      .addUse(VLen);

  B(InitBB, AArch64::UBFMXri)
      .addDef(Tail0)
      .addUse(VLen)
      .addImm(61)
      .addImm(60);

  B(InitBB, AArch64::MSUBXrrr)
      .addDef(Tail1)
      .addUse(Loop1)
      .addUse(Tail0)
      .addUse(SizeReg);

  // SUBSXri source must be GPR64sp: copy Loop1 (GPR64) -> tmp (GPR64sp)
  {
    Register Loop1sp = New64sp();
    B(InitBB, TargetOpcode::COPY).addDef(Loop1sp).addUse(Loop1);
    B(InitBB, AArch64::SUBSXri)
        .addDef(AArch64::XZR)
        .addUse(Loop1sp)
        .addImm(0)
        .addImm(0);
  }

  B(InitBB, AArch64::Bcc)
      .addImm(AArch64CC::EQ)
      .addMBB(TailBB);

  /* -------- Fast loop -------- */
  /* -------- Fast loop -------- */

    unsigned StoreOpc = IsNT ? AArch64::STNT1B_ZRI : AArch64::ST1B_IMM;

    // Address PHIs must be GPR64sp (ST1B_IMM requires GPR64sp)
    Register DstCur  = New64sp();
    Register LoopCur = New64();

    B(FastBB, TargetOpcode::PHI)
        .addDef(DstCur)
        .addUse(Dst0).addMBB(InitBB)
        .addUse(DstCur).addMBB(FastBB);  // placeholder, fixed below

    B(FastBB, TargetOpcode::PHI)
        .addDef(LoopCur)
        .addUse(Loop1).addMBB(InitBB)
        .addUse(LoopCur).addMBB(FastBB); // placeholder, fixed below

    for (int i = 0; i < Unroll; ++i)
      B(FastBB, StoreOpc)
          .addUse(Vec)
          .addUse(Pred)
          .addUse(DstCur)   // GPR64sp — correct for ST1B_IMM
          .addImm(i);

    // INCB needs GPR64: copy GPR64sp -> GPR64, increment, copy back
    Register DstCur64   = New64();
    Register DstNext64  = New64();
    Register DstNext    = New64sp();

    B(FastBB, TargetOpcode::COPY).addDef(DstCur64).addUse(DstCur);
    B(FastBB, AArch64::INCB_XPiI)
        .addDef(DstNext64)
        .addUse(DstCur64)
        .addImm(EffectiveVL)
        .addImm(0);
    B(FastBB, TargetOpcode::COPY).addDef(DstNext).addUse(DstNext64);

    Register LoopNext = New64();

    {
      Register LoopCursp = New64sp();
      B(FastBB, TargetOpcode::COPY).addDef(LoopCursp).addUse(LoopCur);
      B(FastBB, AArch64::SUBSXri)
          .addDef(LoopNext)
          .addUse(LoopCursp)
          .addImm(1)
          .addImm(0);
    }

    B(FastBB, AArch64::Bcc)
        .addImm(AArch64CC::NE)
        .addMBB(FastBB);

    B(FastBB, AArch64::B).addMBB(TailBB);

    // Fix PHI back-edges
    {
      auto PhiIt = FastBB->begin();
      MachineInstr &DstPhi  = *PhiIt++;
      DstPhi.getOperand(3).setReg(DstNext);
      MachineInstr &LoopPhi = *PhiIt;
      LoopPhi.getOperand(3).setReg(LoopNext);
    }

    /* -------- Tail loop -------- */

    // Address PHI must be GPR64sp for ST1B_IMM
    Register DstTailCur = New64sp();
    Register TailCur    = New64();

    B(TailBB, TargetOpcode::PHI)
        .addDef(DstTailCur)
        .addUse(Dst0).addMBB(InitBB)
        .addUse(DstNext).addMBB(FastBB)
        .addUse(DstTailCur).addMBB(TailBB); // placeholder, fixed below

    B(TailBB, TargetOpcode::PHI)
        .addDef(TailCur)
        .addUse(Tail1).addMBB(InitBB)
        .addUse(Tail1).addMBB(FastBB)
        .addUse(TailCur).addMBB(TailBB);    // placeholder, fixed below

    Register Pred2 = NewP();

    B(TailBB, AArch64::WHILELO_PXX_B)
        .addDef(Pred2)
        .addReg(AArch64::XZR)
        .addUse(TailCur);

    B(TailBB, StoreOpc)
        .addUse(Vec)
        .addUse(Pred2)
        .addUse(DstTailCur)  // GPR64sp — correct for ST1B_IMM
        .addImm(0);

    // INCB needs GPR64: copy GPR64sp -> GPR64, increment, copy back
    Register DstTailCur64  = New64();
    Register DstTailNext64 = New64();
    Register DstTailNext   = New64sp();

    B(TailBB, TargetOpcode::COPY).addDef(DstTailCur64).addUse(DstTailCur);
    B(TailBB, AArch64::INCB_XPiI)
        .addDef(DstTailNext64)
        .addUse(DstTailCur64)
        .addImm(AArch64SVEPredPattern::vl1)
        .addImm(0);
    B(TailBB, TargetOpcode::COPY).addDef(DstTailNext).addUse(DstTailNext64);

    Register TailNext = New64();

    B(TailBB, AArch64::DECB_XPiI)
        .addDef(TailNext)
        .addUse(TailCur)
        .addImm(AArch64SVEPredPattern::vl1)
        .addImm(0);

    {
      Register TailNextsp = New64sp();
      B(TailBB, TargetOpcode::COPY).addDef(TailNextsp).addUse(TailNext);
      B(TailBB, AArch64::SUBSXri)
          .addDef(AArch64::XZR)
          .addUse(TailNextsp)
          .addImm(0)
          .addImm(0);
    }

    B(TailBB, AArch64::Bcc)
        .addImm(AArch64CC::GT)
        .addMBB(TailBB);

    B(TailBB, AArch64::B).addMBB(ExitBB);

    // Fix TailBB PHI back-edges
    {
      auto PhiIt = TailBB->begin();
      MachineInstr &DstPhi  = *PhiIt++;
      DstPhi.getOperand(5).setReg(DstTailNext);
      MachineInstr &TailPhi = *PhiIt;
      TailPhi.getOperand(5).setReg(TailNext);
    }

  MI.eraseFromParent();

  MBB.updateTerminator(ExitBB);
  return true;
}

FunctionPass *llvm::createAArch64SVE2MemOpExpandPass() {
  return new AArch64SVE2MemOpExpand();
}
