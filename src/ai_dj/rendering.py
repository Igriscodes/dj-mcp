from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
from ai_dj.config import DJConfig, get_default_config
from ai_dj.errors import DJError, ValidationError
from ai_dj.mixer import Mixer
from ai_dj.timeline import Timeline
from ai_dj.validation import validate_disk_space_for_render


class ProjectRenderer:
    def __init__(self, config: Optional[DJConfig] = None):
        self.config = config or get_default_config()

    def render_timeline(
        self,
        timeline: Timeline,
        mixer: Mixer,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        sr = sample_rate or self.config.sample_rate
        dur = (end_sec - start_sec) if end_sec else timeline.total_duration_sec
        est_mb = max(10, int(dur * sr * 2 * 4 / (1024 * 1024)))
        validate_disk_space_for_render(est_mb, self.config)

        # 1. Render all tracks into individual channel buffers
        channel_signals: Dict[str, np.ndarray] = {}
        for track_id, track in timeline.tracks.items():
            track_audio = track.render_track_audio(
                target_sample_rate=sr,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            ch_id = track.channel_id
            if ch_id in channel_signals:
                # Add to existing channel buffer if multiple tracks share a channel
                cur = channel_signals[ch_id]
                max_l = max(cur.shape[1], track_audio.shape[1])
                combined = np.zeros((2, max_l), dtype=np.float32)
                combined[:, :cur.shape[1]] += cur
                combined[:, :track_audio.shape[1]] += track_audio
                channel_signals[ch_id] = combined
            else:
                channel_signals[ch_id] = track_audio

        # 2. Mix channel signals through mixer
        master_mix = mixer.mix_signals(channel_signals, sr)
        return master_mix
