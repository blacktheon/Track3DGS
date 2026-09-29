"""Run the pinned production trainer with a bounded, route-only image cache."""
from collections import OrderedDict
from collections.abc import Sequence
import json
import os
import sys
import time


class BoundedImageCache(Sequence):
    def __init__(self, count, loader, capacity=32):
        if capacity < 1: raise ValueError('Cache capacity must be positive')
        self.count, self.loader, self.capacity = count, loader, capacity
        self.cache = OrderedDict()

    def __len__(self): return self.count

    @property
    def resident_count(self): return len(self.cache)

    def __getitem__(self, index):
        if index < 0: index += self.count
        if not 0 <= index < self.count: raise IndexError(index)
        if index not in self.cache:
            while len(self.cache) >= self.capacity: self.cache.popitem(last=False)
            self.cache[index] = self.loader(index)
        self.cache.move_to_end(index)
        return self.cache[index]


def install_bounded_cache():
    import importlib.metadata
    import torch
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
    if importlib.metadata.version('nerfstudio') != '1.1.5':
        raise RuntimeError('Route cache adapter is validated for Nerfstudio1.1.5 only')

    def load_images(self, split, cache_images_device):
        if cache_images_device != 'cpu': raise ValueError('Route cache must be CPU')
        dataset = self.train_dataset if split == 'train' else self.eval_dataset
        distortion = dataset.cameras.distortion_params
        if distortion is not None and not torch.all(distortion == 0):
            raise ValueError('Route adapter requires already projected undistorted pinhole views')
        def load(index):
            data = dataset.get_data(index, image_type=self.config.cache_images_type)
            camera = dataset.cameras[index].reshape(())
            if data['image'].shape[:2] != (camera.height.item(), camera.width.item()):
                raise ValueError('Image dimensions disagree with route camera')
            return data
        self.train_cameras = self.train_dataset.cameras
        capacity = int(os.environ.get('TRACK3DGS_IMAGE_CACHE', '32')) if split == 'train' else 8
        print(f'ROUTE_CACHE {split}: {len(dataset)} views, at most {capacity} decoded images, no pinned cache', flush=True)
        return BoundedImageCache(len(dataset), load, capacity)
    FullImageDatamanager._load_images = load_images


def main():
    import torch
    import psutil
    install_bounded_cache()
    action = sys.argv.pop(1)
    started = time.monotonic()
    if action == 'train':
        from nerfstudio.engine.trainer import Trainer
        original = Trainer.train_iteration
        def iteration(self, step):
            result = original(self, step)
            if step % 1000 == 0 or step + 1 == self._start_step + self.config.max_num_iterations:
                print('ROUTE_PROGRESS ' + json.dumps({'step': step, 'loss': float(result[0].detach()),
                    'splats': int(self.pipeline.model.num_points), 'elapsed_seconds': time.monotonic()-started,
                    'gpu_peak_allocated_gb': torch.cuda.max_memory_allocated()/2**30,
                    'process_ram_gb': psutil.Process().memory_info().rss/2**30}), flush=True)
            return result
        Trainer.train_iteration = iteration
        from nerfstudio.scripts.train import entrypoint
    elif action == 'export':
        from nerfstudio.scripts.exporter import entrypoint
    else:
        raise ValueError('Expected train or export')
    entrypoint()


if __name__ == '__main__': main()
