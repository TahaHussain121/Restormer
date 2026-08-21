"""Peak training-step VRAM for ONE arm. One arm per process, deliberately.

Measuring two arms in a single process does not work: the first arm's optimizer
still holds references to its parameters, so `del net` frees nothing, and the
second measurement inherits the first one's allocations. That is what exhausted
a 32 GB V100 in the first crossattn smoke run (job 1783386) -- a defect in the
measurement, not in the arm. Run this once per config and compare the printed
numbers; the sbatch wrapper does exactly that, on the a100 the arms train on.
"""
import argparse, copy, json, sys, torch, yaml
sys.path.insert(0, '/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer')
sys.path.insert(0, '/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer/'
                   'dino_analysis_phases/phase3_restoration/scripts')
from basicsr.models.archs import define_network            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--crop', type=int, default=128)
ap.add_argument('--steps', type=int, default=3)
ap.add_argument('--seed', type=int, default=100)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
torch.manual_seed(a.seed)
net = define_network(copy.deepcopy(cfg['network_g'])).cuda().train()
if hasattr(net, 'set_dino_mode'):
    net.set_dino_mode('train128')
ch = 2 if cfg['network_g'].get('dino_source') == 'render' else 1
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4)
torch.manual_seed(0)
x = torch.rand(a.batch, ch, a.crop, a.crop, device='cuda')
gt = torch.rand(a.batch, 1, a.crop, a.crop, device='cuda')
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
for _ in range(a.steps):
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.l1_loss(net(x), gt).backward()
    opt.step()
torch.cuda.synchronize()
print(json.dumps({'config': a.config, 'name': cfg['name'],
                  'batch': a.batch, 'crop': a.crop,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                  'gpu': torch.cuda.get_device_name(0)}))
