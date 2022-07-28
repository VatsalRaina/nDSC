from data_load import get_val_dataloader
import numpy as np

if __name__ == '__main__':
    flair_path = '/home/meri/data/canonical/train_isovox'
    gts_path = '/home/meri/data/canonical/train_isovox'
    data_loader = get_val_dataloader(flair_path, gts_path, num_workers=10, cache_rate=0.1, bm_path=None)

    bg_to_fg_fracs = []
    for data in data_loader:
        target = data['label'].cpu().numpy()
        bg_count = np.sum(target == 0)
        fg_count = np.sum(target == 1)
        bg_to_fg_fracs.append(bg_count / fg_count)

    print(f"Median fraction of number background voxels to foreground voxels: {np.median(bg_to_fg_fracs)}")