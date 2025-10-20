"""Lightweight training/eval logger: console, CSV, and TensorBoard (scalars, histograms, videos)."""


import os
import csv
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from termcolor import colored
from collections import defaultdict

# ------------------------
# Display / CSV formats
# ------------------------

COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('duration', 'D', 'time'),
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
]

# Match the keys actually logged by your SACAgent
AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward',   'BR',    'float'),
        ('actor_loss',     'ALOSS', 'float'),
        ('critic_loss',    'CLOSS', 'float'),
        ('alpha_loss',     'TLOSS', 'float'),
        ('alpha',          'ALPHA', 'float'),  # was alpha_value
        ('entropy',        'ENT',   'float'),  # was actor_entropy
        ('target_entropy', 'TENT',  'float'),  # optional but useful
    ]
}


# ------------------------
# Helpers
# ------------------------

class AverageMeter(object):
    # tracks a running average of values (sum/count).
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, value, n=1):
        self._sum += float(value)
        self._count += int(n)

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    # aggregates meters, formats output, and writes logs to CSV and console.
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, 'csv')
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        # newline='' avoids extra blank lines on Windows
        self._csv_file = open(self._csv_file_name, 'w', newline='', encoding='utf-8')
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f'{prefix}.{suffix}'
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
            except Exception:
                # ignore removal errors
                pass
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for key, meter in self._meters.items():
            if key.startswith('train'):
                k = key[len('train') + 1:]
            else:
                k = key[len('eval') + 1:]
            k = k.replace('/', '_')
            data[k] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=sorted(data.keys()),
                restval=0.0
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            return f'{key}: {int(value)}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise ValueError(f'invalid format type: {ty}')

    def _dump_to_console(self, data, prefix):
        prefix_col = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix_col: <14}']
        for key, disp_key, ty in self._formating:
            val = data.get(key, 0)
            pieces.append(self._format(disp_key, val, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if not self._meters:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()

    def close(self):
        try:
            if hasattr(self, "_csv_file") and self._csv_file and not self._csv_file.closed:
                self._csv_file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ------------------------
# Public Logger
# ------------------------

class Logger(object):
    # public logging API that records train/eval metrics to TensorBoard, CSV, and console with interval control.
    def __init__(self, log_dir, save_tb=False, log_frequency=10000, agent='sac'):
        self._log_dir = log_dir
        self._log_frequency = int(log_frequency)

        # TensorBoard
        self._sw = None
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except Exception:
                    print("logger.py warning: Unable to remove tb directory")
            self._sw = SummaryWriter(tb_dir)

        # per-agent train format
        if agent not in AGENT_TRAIN_FORMAT:
            raise ValueError(f"Unknown agent '{agent}' for logging format.")
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]

        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'), formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'), formating=COMMON_EVAL_FORMAT)

    def _should_log(self, step, log_frequency=None):
        freq = int(log_frequency or self._log_frequency)
        return step % max(1, freq) == 0

    # ------------- TB helpers -------------

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, float(value), step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.as_tensor(frames)
            if frames.ndim == 4:
                frames = frames.unsqueeze(0)  # (1, T, C, H, W)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    # ------------- Public API -------------

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval'), "log key must start with 'train' or 'eval'"
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._try_sw_log(key, value / max(1, n), step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if getattr(param.weight, 'grad', None) is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if getattr(param.bias, 'grad', None) is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        else:
            raise ValueError(f'invalid log type: {ty}')

    # ------------- Cleanup -------------

    def close(self):
        try:
            if self._sw is not None:
                self._sw.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_train_mg") and self._train_mg is not None:
                self._train_mg.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_eval_mg") and self._eval_mg is not None:
                self._eval_mg.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
