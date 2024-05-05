

import os
import torch
import torch.nn.functional as F
import open3d as o3d
import seaborn as sns
import numpy.random as rand
from scipy.spatial.transform import Rotation


smoothL1Loss = torch.nn.SmoothL1Loss()
mseL2Loss = torch.nn.MSELoss()
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
def rotate_pcd(pcd, R = Rotation.from_matrix([[0,1,0], [1, 0, 0], [0, 0, 1]])):

    return (R.apply(pcd))

def un_rotate_pcd(pcd, R = Rotation.from_matrix([[0,1,0], [1, 0, 0], [0, 0, 1]])):

    return (R.inv().apply(pcd))
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
def create_rand_rotation():
    R= Rotation.random().as_euler('xyz')

    return R

#loss parameters based on sc3k general loss function


#weights configured according to sc3k paper with exception of centroid which is a new loss added by us
'''
weight_separation = 0.5
weight_centroid = 2
weight_shape = 6
weight_volume = 1
weight_consistency = 1
weight_overlap = 0.5
weight_pose = 0.05
'''

#weights configured according to sc3k paper with exception of centroid which is a new loss added by us
'''
weight_separation = 0.5
weight_centroid = 2
weight_shape = 6
weight_volume = 1
weight_consistency = 1
weight_overlap = 0.5
weight_pose = 0.05
'''
weight_separation = 1.5
weight_centroid = 1
weight_shape = 7
weight_volume = 1
weight_consistency = 1
weight_equidistance = 0.01
weight_overlap = 0.5
weight_pose = 0.05

def compute_loss_pr(kp1, kp2, data, writer, step, cfg, split='split??'):
    #device calls heavily modeled after sc3k code to avoid cuda errors
    device = kp1.device
    loss_centroid1 = centroid_loss(kp1, data[0].float().to(device))
    loss_centroid2 = centroid_loss(kp2, data[2].float().to(device))
    loss_separation1 = separation_loss_pr(kp1)
    loss_separation2 = separation_loss_pr(kp2)
    loss_shape1 = shape_loss_pr(kp1, data[0].float().to(device))
    loss_shape2 = shape_loss_pr(kp2, data[2].float().to(device)) 
    loss_volume1 = volume_loss_pr(kp1, data[0].float().to(device))    
    loss_volume2 = volume_loss_pr(kp2, data[2].float().to(device))  

    loss_overlap1 = average_overlap_loss(kp1) * weight_overlap
    loss_overlap2 = average_overlap_loss(kp2) * weight_overlap     
    loss_consistency = consistency_loss_pr(kp1, data[1].float().to(device), kp2, data[3].float().to(device))    
    loss_pose = pose_loss(kp1, kp2, data[1].float().to(device), data[3].float().to(device))

    loss_equidistance1 = equidistance_loss(kp1)
    loss_equidistance2 = equidistance_loss(kp2)
    individual_loss = (loss_overlap1 + loss_overlap2 +loss_centroid1*weight_centroid+ loss_centroid2*weight_centroid+
                       loss_separation1 * weight_separation + loss_separation2 * weight_separation +
                       loss_shape1 * weight_shape+loss_shape2 *weight_shape + 
                       loss_volume1*weight_volume+loss_volume2*weight_volume + loss_equidistance1*weight_equidistance+loss_equidistance2*weight_equidistance)
    mutual_dependency_loss = loss_consistency * weight_consistency + loss_pose * weight_pose
    total_loss = individual_loss+mutual_dependency_loss


    return total_loss

def average_overlap_loss(kp, tau = 0.05):
    distances = torch.cdist(kp, kp)
    # we subtract all the self comparisons (so size of batch times size of keypoints)
    # use 1 and 0 for Iverson bracket.  subtract 1 per row for self comparison
    loss_matrix = torch.sum(torch.sum(torch.where(distances<tau, 1, 0), 0), 0) -len(kp)
    loss = torch.mean(loss_matrix/(len(kp)**2)) #get the mean across batches

    return loss

    

def consistency_loss_pr(kp1, rotation1, kp2, rotation2):
    #note: I needed to reference original function for rotation axis ordering (xyz vs xzy)
    rotation1 = torch.permute(rotation1, [0,2,1])
    rotation2 = torch.permute(rotation2, [0,2,1])          
    rotated_kp1 = torch.matmul(kp1, rotation1)
    rotated_kp2  = torch.matmul(kp2, rotation2)
    #diff_rotated = rotated_kp1-rotated_kp2
    loss = mseL2Loss(rotated_kp1, rotated_kp2)

    return(loss)


def separation_loss_pr(kp):
    #Separation loss: Are our keypoints as spread out as they can be from each other?  
    #kp shape is [26, 10, 3]

    #distance shape is 26, 10, 10


    distances = torch.cdist(kp, kp)    
    smallest_distances = torch.topk(distances, k= 2, dim=1, largest= False)[0]
    smallest_distances= torch.sum(smallest_distances, dim=1)
    
    loss = torch.mean(smallest_distances)
    if (loss < 0.01):
        loss = 0.01

    loss=1/loss

    return loss

def shape_loss_pr(kp, pcd):
    #Shape loss: How far are we from the actual object    
    #kp shape is [26, 10, 3]

    #distance shape is 26, 10, 2048
    distances = torch.cdist(kp, pcd)
    smallest_distances = torch.topk(distances, k= 1, largest= False)   

    loss=torch.mean(smallest_distances[0]) #0 because there is a values vector and an indices vector
     
    return loss

def equidistance_loss(kp, tau=0.05):
    #Shape loss: favor more equal spacing of keypoints    
    #kp shape is [26, 10, 3]

    #distance shape is 26, 10, 2048
    distances = torch.cdist(kp, kp)

    differences = torch.cdist(distances, distances)
    loss_matrix = torch.sum(torch.sum(torch.where(differences<tau, 1, 0), 0), 0)  

    loss = 1 /loss_matrix

    return torch.mean(loss)

def volume_loss_pr(kp, pcd):
    #volume loss: bounding box comparison
    #kp shape is [26, 10, 3]
    min_kp = torch.min(kp, dim=1)[0] #[0] because we don't need indices only coordinates
    max_kp = torch.max(kp, dim=1)[0]
    size_kp = max_kp-min_kp
    vol_kp = torch.prod(size_kp, 1)

    min_pcd = torch.min(pcd, dim=1)[0]
    max_pcd = torch.max(pcd, dim=1)[0]
    size_pcd = max_pcd-min_pcd

    vol_pcd = torch.prod(size_pcd, 1)
    #initially did volume as I interpreted in the paper but this made the losses too large
    #smooth L1 loss as used by the sc3k paper
    #official documentation https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

    loss = smoothL1Loss(size_kp, size_pcd)

    return loss


def centroid_loss(pc, kp):
    #Centroid loss: How far the centroid of the keypoints is from the centroid of the pointcloud (MSE)
    loss = mseL2Loss(torch.mean(pc, dim=1), torch.mean(kp, dim=1))

    return loss




#Citing the last two function from SC3K code
# https://github.com/iit-pavis/sc3k

def pose_loss(kp1, kp2, rot1, rot2):
    '''

    Parameters
    ----------
    kp1     Estimated key-points 1
    kp2     Transformed version of the estimated key-points 2
    rot1    pose of KP1
    rot2    pose of KP2

    rot     GT relative pose b/w kp1 and kp2

    Returns     Loss => Error in relative pose b/w kp1 and kp2 [Forbunius Norm]
    -------

    '''
    device = kp1.device
    gt_rot = torch.bmm(rot1.double(), torch.transpose(rot2.double(), 1, 2))
    mat = batch_compute_similarity_transform_torch(kp1, kp2)
    # mat = batch_compute_similarity_transform_torch(kp1.permute(0, 2, 1), kp2.permute(0, 2, 1))
    frob = torch.sqrt(torch.sum(torch.square(gt_rot - mat)))    # Forbunius Norm

    angle_ = torch.mean(torch.arcsin(
        torch.clamp(torch.min(torch.tensor(1.).to(device), frob / (2. * torch.sqrt(torch.tensor(2.).to(device)))), -0.99999,
                    0.99999)))

    return angle_



def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    help: https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427

    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))      # position
    R = torch.linalg.inv(R)                 # rotation

    #
    # # 5. Recover scale.
    # scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    #
    # # 6. Recover translation.
    # t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
    #
    # # 7. Error:
    # S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    #
    # if transposed:
    #     S1_hat = S1_hat.permute(0,2,1)
    #
    # return S1_hat

    return R