# 这是我用来设置攻击产生的对抗样本的存放的设置
TPATH = r"F:\csy\ASVNormClip64600Dataset\attackdataset3"
TLABEL = r"F:\csy\ASVNormClip64600Dataset\attackdataset3_label.txt"
VMIPATH = r"F:\csy\ASVNormClip40000Dataset\attackdataset1"
VMILABEL = r"F:\csy\ASVNormClip40000Dataset\attackdataset1_label.txt"

# rawnet726
RawNet726_model = r"F:\Adversarial-1D\ALLmodels\RAWNET72600\epoch_28_99.7_99.3.pth"
RawNet726_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw726\FGSM"
RawNet726_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw726\PGD"
RawNet726_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw726\CW"
RawNet726_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw726\Deepfool"
RawNet726_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw726\MIFGSM"
RawNet726_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw726\SMIFGSM"

# rawnet646
RawNet646_model = r"F:\Adversarial-1D\ALLmodels\RAWNET64600\epoch_29_99.7_99.5.pth"
RawNet646_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw646\FGSM"
RawNet646_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw646\PGD"
RawNet646_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw646\CW"
RawNet646_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw646\Deepfool"
RawNet646_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw646\MIFGSM"
RawNet646_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw646\SMIFGSM"

# rawnet56
RawNet56_model = r"F:\Adversarial-1D\ALLmodels\RAWNET56000\model_30_32_1024_lr0.0003_norm\epoch_26_99.6_99.5.pth"
RawNet56_model_2 = r"F:\Adversarial-1D\ALLmodels\RAWNET56000\model_30_32_1024_lr0.0003_norm\epoch_25_99.7_99.4.pth"
RawNet56_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\FGSM"
RawNet56_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\PGD"
RawNet56_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\CW"
RawNet56_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\Deepfool"
RawNet56_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\MIFGSM"
RawNet56_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\SMIFGSM"
RawNet56_Satics_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\Satics"
RawNet56_Satics_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\SaticsCW"
RawNet56_Subens_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\SubensCW"
RawNet56_Subens_sub_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw56\SubensCW_sub"

# rawnet48
RawNet48_model = r"F:\Adversarial-1D\ALLmodels\RAWNET48000\epoch_24_99.2_97.8.pth"
RawNet48_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw48\FGSM"
RawNet48_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw48\PGD"
RawNet48_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw48\CW"
RawNet48_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw48\Deepfool"
RawNet48_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw48\MIFGSM"

# rawnet4
RawNet4_model = r"F:\Adversarial-1D\ALLmodels\RAWNET40000\epoch_26_99.4_98.0.pth"
RawNet4_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw4\FGSM"
RawNet4_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw4\PGD"
RawNet4_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw4\CW"
RawNet4_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw4\Deepfool"
RawNet4_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\Raw4\MIFGSM"

Attacks_Name = ["CW",
                "DeepFool",
                "FGSM",
                "MIFGSM",
                "PGD",
                # "SMIFGSM"
                "Satics"
                ]
RawNet726_Advset = [RawNet726_CWAdvsetPath,
                    RawNet726_DeepfoolAdvsetPath,
                    RawNet726_FGSMAdvsetPath,
                    RawNet726_MIFGSMAdvsetPath,
                    RawNet726_PGDAdvsetPath,
                    RawNet726_SMIFGSMAdvsetPath
                    ]
RawNet646_Advset = [RawNet646_CWAdvsetPath,
                    RawNet646_DeepfoolAdvsetPath,
                    RawNet646_FGSMAdvsetPath,
                    RawNet646_MIFGSMAdvsetPath,
                    RawNet646_PGDAdvsetPath,
                    RawNet646_SMIFGSMAdvsetPath
                    ]
RawNet56_Advset = [RawNet56_CWAdvsetPath,
                   RawNet56_DeepfoolAdvsetPath,
                   RawNet56_FGSMAdvsetPath,
                   RawNet56_MIFGSMAdvsetPath,
                   RawNet56_PGDAdvsetPath,
                   # RawNet56_SMIFGSMAdvsetPath,
                   RawNet56_Satics_MIFGSMAdvsetPath,
                   ]
RawNet48_Advset = [RawNet48_CWAdvsetPath,
                   RawNet48_DeepfoolAdvsetPath,
                   RawNet48_FGSMAdvsetPath,
                   RawNet48_MIFGSMAdvsetPath,
                   RawNet48_PGDAdvsetPath
                   ]
RawNet4_Advset = [RawNet4_CWAdvsetPath,
                  RawNet4_DeepfoolAdvsetPath,
                  RawNet4_FGSMAdvsetPath,
                  RawNet4_MIFGSMAdvsetPath,
                  RawNet4_PGDAdvsetPath
                  ]

# np save path
NpPath = r"F:\Adversarial-1D\NpSave"

# lcnn4
LCNN4_model = r"F:\Adversarial-1D\ALLmodels\LCNN40000\WCE_30_32_0.0001\epoch_28_99.8_99.6.pth"
LCNN4_model_2 = r"F:\Adversarial-1D\ALLmodels\LCNN40000\WCE_30_32_0.0001\epoch_29_97.7_99.5.pth"
LCNN4_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\FGSM"
LCNN4_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\PGD"
LCNN4_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\CW"
LCNN4_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\Deepfool"
LCNN4_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\MIFGSM"
LCNN4_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\SMIFGSM"
LCNN4_Satics_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\Satics"
LCNN4_Satics_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\SaticsCW"
LCNN4_Subens_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\SubensCW"
LCNN4_Subens_sub_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN4\SubensCW_sub"

# lcnn48
LCNN48_model = r"F:\Adversarial-1D\ALLmodels\LCNN48000\WCE_30_32_0.0001\epoch_28_99.6_99.9.pth"
LCNN48_model_2 = r"F:\Adversarial-1D\ALLmodels\LCNN48000\WCE_30_32_0.0001\epoch_29_99.8_99.8.pth"
LCNN48_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\FGSM"
LCNN48_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\PGD"
LCNN48_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\CW"
LCNN48_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\Deepfool"
LCNN48_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\MIFGSM"
LCNN48_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\SMIFGSM"
LCNN48_Satics_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\Satics"
LCNN48_Satics_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\SaticsCW"
LCNN48_Subens_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\SubensCW"
LCNN48_Subens_sub_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN48\SubensCW_sub"

# lcnn56
LCNN56_model = r"F:\Adversarial-1D\ALLmodels\LCNN56000\WCE_30_32_0.0001\epoch_19_99.9_99.9.pth"
LCNN56_model_2 = r"F:\Adversarial-1D\ALLmodels\LCNN56000\WCE_30_32_0.0001\epoch_26_99.9_99.8.pth"
LCNN56_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\FGSM"
LCNN56_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\PGD"
LCNN56_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\CW"
LCNN56_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\Deepfool"
LCNN56_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\MIFGSM"
LCNN56_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\SMIFGSM"
LCNN56_Satics_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\Satics"
LCNN56_Satics_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\SaticsCW"
LCNN56_Subens_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\SubensCW"
LCNN56_Subens_sub_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN56\SubensCW_sub"

# lcnn646
LCNN646_model = r"F:\Adversarial-1D\ALLmodels\LCNN646000\WCE_30_32_0.0001\epoch_13_99.7_99.7.pth"
LCNN646_model_2 = r"F:\Adversarial-1D\ALLmodels\LCNN646000\WCE_30_32_0.0001\epoch_14_99.6_99.8.pth"
LCNN646_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\FGSM"
LCNN646_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\PGD"
LCNN646_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\CW"
LCNN646_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\Deepfool"
LCNN646_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\MIFGSM"
LCNN646_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\SMIFGSM"
LCNN646_Satics_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\Satics"
LCNN646_Satics_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\SaticsCW"
LCNN646_Subens_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\SubensCW"
LCNN646_Subens_sub_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN646\SubensCW_sub"

# lcnn726
LCNN726_model = r"F:\Adversarial-1D\ALLmodels\LCNN726000\WCE_30_16_0.0001\epoch_13_99.8_99.8.pth"
LCNN726_model_2 = r"F:\Adversarial-1D\ALLmodels\LCNN726000\WCE_30_16_0.0001\epoch_4_99.6_99.7.pth"
LCNN726_FGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\FGSM"
LCNN726_PGDAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\PGD"
LCNN726_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\CW"
LCNN726_DeepfoolAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\Deepfool"
LCNN726_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\MIFGSM"
LCNN726_SMIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\SMIFGSM"
LCNN726_Satics_MIFGSMAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\Satics"
LCNN726_Satics_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\SaticsCW"
LCNN726_Subens_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\SubensCW"
LCNN726_Subens_sub_CWAdvsetPath = r"F:\Adversarial-1D\AavSave\LCNN726\SubensCW_sub"
