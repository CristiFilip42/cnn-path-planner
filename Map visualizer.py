import torch,sys,types,plt

class MapSample:
    pass

mod = types.ModuleType("dataset.map_sample")
mod.MapSample = MapSample
sys.modules["dataset"] = types.ModuleType("dataset")
sys.modules["dataset.map_sample"] = mod

s = torch.load("Dataset/map_dataset/train/0a0a11ac-9c44-48db-87f0-2377cf26a14f.pt", weights_only = False )

print(type(s))
print(vars(s).keys())

print(s.map.shape, s.map.dtype)
print(s.start, s.goal)
print(s.path.shape)

grid = s.map.numpy()
path = s.path.numpy()

plt.imshow(grid, cmap="gray_r")
plt.show()