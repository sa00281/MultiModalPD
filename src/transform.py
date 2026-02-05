import torch

class Norm2D(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        mean = torch.mean(sample, dim=0)
        std = torch.std(sample, dim=0)

        epsilon = 1e-8
        std = torch.where(std == 0, torch.tensor(epsilon), std)

        normalized_sample = (sample - mean) / std

        return normalized_sample


class RandomTranslation(object):
    def __init__(self,
                 range: list = [-0.01, 0.01]):        
        self.range = range

    def __call__(self, sample):
        translate = torch.eye(3)

        t_x = torch.zeros(1).uniform_(self.range[0], self.range[1])
        t_y = torch.zeros(1).uniform_(self.range[0], self.range[1])
        t_z = torch.zeros(1).uniform_(self.range[0], self.range[1])

        translate[0, 0] = translate[0, 0] + t_x
        translate[1, 1] = translate[1, 1] + t_y
        translate[2, 2] = translate[2, 2] + t_z

        return torch.matmul(sample, translate)


class RandomRotation(object):
    def __init__(self,
                 range: list = [-0.3, 0.3]):        
        self.range = range

    def rotation_matrix(self, rotation):
        cos_r, sin_r = rotation.cos(), rotation.sin()
        zeros = torch.zeros(rotation.shape[0], 1)
        ones = torch.ones(rotation.shape[0], 1)

        r1 = torch.stack((ones, zeros, zeros), dim=-1)
        rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim=-1)
        rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim=-1)
        rx = torch.cat((r1, rx2, rx3), dim=1)

        ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim=-1)
        r2 = torch.stack((zeros, ones, zeros), dim=-1)
        ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim=-1)
        ry = torch.cat((ry1, r2, ry3), dim=1)

        rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim=-1)
        rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3], zeros), dim=-1)
        r3 = torch.stack((zeros, zeros, ones), dim=-1)
        rz = torch.cat((rz1, rz2, r3), dim=1)

        return torch.matmul(rz, torch.matmul(ry, rx))


    def __call__(self, sample):
        T, _, _ = sample.shape

        data = sample.permute(0, 2, 1) # [T, C, V]

        rotation = torch.zeros(3).uniform_(self.range[0], self.range[1])
        rotation = torch.stack([rotation, ] * T, dim=0)
        rotation = self.rotation_matrix(rotation)
        data = torch.matmul(rotation, data)
        
        data = data.permute(0, 2, 1) # [T, V, C]

        return data
 
    
class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)