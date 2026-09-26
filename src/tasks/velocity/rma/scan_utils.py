"""Helpers for height-scan (scandot) sizing — matches mjlab GridPatternCfg."""

from __future__ import annotations


def grid_pattern_num_rays(size: tuple[float, float], resolution: float) -> int:
  """Number of rays produced by ``GridPatternCfg.generate_rays`` for ``size``/``resolution``."""
  size_x, size_y = size
  # Match torch.arange(-s/2, s/2 + res*0.5, res) length without importing torch.
  def _n(half_extent: float) -> int:
    # Inclusive walk with the same float endpoint rule as torch.arange.
    count = 0
    x = -half_extent
    end = half_extent + resolution * 0.5
    # Guard against float drift; same formula as extreme-parkour grid count.
    while x < end - 1e-9:
      count += 1
      x += resolution
    return count

  return _n(size_x / 2) * _n(size_y / 2)


def terrain_scan_num_rays_from_cfg(env_cfg) -> int:
  """Read ``terrain_scan`` GridPattern and return ray count (0 if missing)."""
  for sensor in env_cfg.scene.sensors or ():
    if getattr(sensor, "name", None) != "terrain_scan":
      continue
    pattern = getattr(sensor, "pattern", None)
    if pattern is None:
      return 0
    size = tuple(pattern.size)
    resolution = float(pattern.resolution)
    return grid_pattern_num_rays(size, resolution)
  return 0
