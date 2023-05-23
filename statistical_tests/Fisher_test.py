import numpy as np
from scipy.stats import chi2_contingency, chisquare, fisher_exact
from scipy.stats import binom_test

list_VGG = [1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0]


list_ResNet = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,
               1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]


list_ResNetVLAD = [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,
                   1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                   1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,
                   1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
                   1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,
                   1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,
                   1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
                   0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,0,
                   1,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,
                   1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1]


list_SIFT = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
             1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


list_ORB = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


list_Proposed = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                 1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,
                 1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,0,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
                 1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


arr_VGG = np.array(list_VGG)
arr_ResNet = np.array(list_ResNet)
arr_ResNetVLAD = np.array(list_ResNetVLAD)
arr_SIFT = np.array(list_SIFT)
arr_ORB = np.array(list_ORB)
arr_Proposed = np.array(list_Proposed)



oddsratio, p = fisher_exact([[118, 27], [42, 133]])
print('TEST VGG-ResNet, p=', p)

oddsratio, p = fisher_exact([[118, 28], [42, 132]])
print('TEST VGG-ResNetVLAD, p=', p)

oddsratio, p = fisher_exact([[118, 128], [42, 32]])
print('TEST VGG-SIFT, p=', p)

oddsratio, p = fisher_exact([[118, 112], [42, 48]])
print('TEST VGG-ORB, p=', p)

oddsratio, p = fisher_exact([[118, 10], [42, 150]])
print('TEST VGG-Proposed, p=', p)



oddsratio, p = fisher_exact([[27, 28], [133, 132]])
print('TEST ResNet-ResNetVLAD, p=', p)

oddsratio, p = fisher_exact([[27, 128], [133, 32]])
print('TEST ResNet-SIFT, p=', p)

oddsratio, p = fisher_exact([[27, 112], [133, 48]])
print('TEST ResNet-ORB, p=', p)

oddsratio, p = fisher_exact([[27, 10], [133, 150]])
print('TEST ResNet-Proposed, p=', p)




oddsratio, p = fisher_exact([[28, 128], [132, 32]])
print('TEST ResNetVLAD-SIFT, p=', p)

oddsratio, p = fisher_exact([[28, 112], [132, 48]])
print('TEST ResNetVLAD-ORB, p=', p)

oddsratio, p = fisher_exact([[28, 10], [132, 150]])
print('TEST ResNetVLAD-Proposed, p=', p)



oddsratio, p = fisher_exact([[128, 112], [32, 48]])
print('TEST SIFT-ORB, p=', p)

oddsratio, p = fisher_exact([[128, 10], [32, 150]])
print('TEST SIFT-Proposed, p=', p)



oddsratio, p = fisher_exact([[112, 10], [48, 150]])
print('TEST ORB-Proposed, p=', p)