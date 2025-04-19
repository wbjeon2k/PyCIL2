def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from pycil2.models.icarl import iCaRL
        return iCaRL(args)
    elif name == "bic":
        from pycil2.models.bic import BiC
        return BiC(args)
    elif name == "podnet":
        from pycil2.models.podnet import PODNet
        return PODNet(args)
    elif name == "lwf":
        from pycil2.models.lwf import LwF
        return LwF(args)
    elif name == "ewc":
        from pycil2.models.ewc import EWC
        return EWC(args)
    elif name == "wa":
        from pycil2.models.wa import WA
        return WA(args)
    elif name == "der":
        from pycil2.models.der import DER
        return DER(args)
    elif name == "finetune":
        from pycil2.models.finetune import Finetune
        return Finetune(args)
    elif name == "replay":
        from pycil2.models.replay import Replay
        return Replay(args)
    elif name == "gem":
        from pycil2.models.gem import GEM
        return GEM(args)
    elif name == "coil":
        from pycil2.models.coil import COIL
        return COIL(args)
    elif name == "foster":
        from pycil2.models.foster import FOSTER
        return FOSTER(args)
    elif name == "rmm-icarl":
        from pycil2.models.rmm import RMM_FOSTER, RMM_iCaRL
        return RMM_iCaRL(args)
    elif name == "rmm-foster":
        from pycil2.models.rmm import RMM_FOSTER, RMM_iCaRL
        return RMM_FOSTER(args)
    elif name == "fetril":
        from pycil2.models.fetril import FeTrIL 
        return FeTrIL(args)
    elif name == "pass":
        from pycil2.models.pa2s import PASS
        return PASS(args)
    elif name == "il2a":
        from pycil2.models.il2a import IL2A
        return IL2A(args)
    elif name == "ssre":
        from pycil2.models.ssre import SSRE
        return SSRE(args)
    elif name == "memo":   
        from pycil2.models.memo import MEMO
        return MEMO(args)
    elif name == "beefiso":
        from pycil2.models.beef_iso import BEEFISO
        return BEEFISO(args)
    elif name == "simplecil":
        from pycil2.models.simplecil import SimpleCIL
        return SimpleCIL(args)
    elif name == "acil":
        from pycil2.models.acil import ACIL
        return ACIL(args)
    elif name == "ds-al":
        from pycil2.models.dsal import DSAL
        return DSAL(args)
    elif name == "aper_finetune":
        from pycil2.models.aper_finetune import APER_FINETUNE
        return APER_FINETUNE(args)
    elif name == "tagfex":
        from pycil2.models.tagfex import TagFex
        return TagFex(args)
    else:
        assert 0
