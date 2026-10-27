from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
from ai_dj.config import DJConfig, get_default_config
from ai_dj.errors import DJError, PrerequisiteError, ValidationError
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
        decks: Optional[Dict[str, Any]] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        sr = sample_rate or self.config.sample_rate

        # Determine duration
        if end_sec is not None:
            dur = max(0.1, end_sec - start_sec)
        elif timeline.total_duration_sec > 0:
            dur = max(0.1, timeline.total_duration_sec - start_sec)
        elif decks:
            loaded_durs = []
            for d in decks.values():
                if d.loaded_audio_path and Path(d.loaded_audio_path).exists():
                    try:
                        loaded_durs.append(sf.info(str(d.loaded_audio_path)).duration)
                    except Exception:
                        pass
            dur = max(loaded_durs) if loaded_durs else 30.0
        else:
            dur = 30.0

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
                cur = channel_signals[ch_id]
                max_l = max(cur.shape[1], track_audio.shape[1])
                combined = np.zeros((2, max_l), dtype=np.float32)
                combined[:, :cur.shape[1]] += cur
                combined[:, :track_audio.shape[1]] += track_audio
                channel_signals[ch_id] = combined
            else:
                channel_signals[ch_id] = track_audio

        # 2. Render DJ decks into mixer channel buffers
        if decks:
            playing_decks = [
                d for d in decks.values()
                if d.state == "playing" and d.loaded_audio_path and Path(d.loaded_audio_path).exists()
            ]
            if not playing_decks and not timeline.tracks:
                decks_to_render = [
                    d for d in decks.values()
                    if d.loaded_audio_path and Path(d.loaded_audio_path).exists()
                ]
            else:
                decks_to_render = playing_decks

            for deck in decks_to_render:
                deck_audio = deck.render_deck_audio(
                    duration_sec=dur,
                    target_sample_rate=sr,
                )
                ch_id = f"ch_{deck.deck_id.replace('deck_', '')}"
                if ch_id not in mixer.channels:
                    if deck.deck_id in mixer.channels:
                        ch_id = deck.deck_id
                    elif mixer.channels:
                        ch_id = list(mixer.channels.keys())[0]

                if ch_id in channel_signals:
                    cur = channel_signals[ch_id]
                    max_l = max(cur.shape[1], deck_audio.shape[1])
                    combined = np.zeros((2, max_l), dtype=np.float32)
                    combined[:, :cur.shape[1]] += cur
                    combined[:, :deck_audio.shape[1]] += deck_audio
                    channel_signals[ch_id] = combined
                else:
                    channel_signals[ch_id] = deck_audio

        if not channel_signals or all(sig.shape[1] == 0 for sig in channel_signals.values()):
            raise PrerequisiteError(
                code="NOTHING_TO_RENDER",
                message="Cannot render audio: timeline contains no audio clips and no DJ decks have loaded audio.",
                required_action="Load audio onto a DJ deck using deck_load, or add clips to the timeline using clip_create.",
                suggested_tool="deck_load",
                suggested_arguments={"deck_id": "deck_a", "track_id": "track_1"},
            )

        # 3. Mix channel signals through mixer
        master_mix = mixer.mix_signals(channel_signals, sr)
        if master_mix.shape[1] == 0:
            raise DJError(
                code="RENDER_OUTPUT_EMPTY",
                message="Rendering resulted in 0 audio samples. Check that audio clips or deck positions contain valid audio.",
            )
        return master_mix
