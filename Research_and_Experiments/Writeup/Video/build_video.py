from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import subprocess
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
WRITEUP = HERE.parent
ROOT = WRITEUP.parents[1]
FRAMES = HERE / 'frames'
AUDIO = HERE / 'audio'
CLIPS = HERE / 'clips'
for folder in [FRAMES, AUDIO, CLIPS]:
    folder.mkdir(exist_ok=True)
NAVY, TEAL, AMBER, GRAY = '#132D40', '#147D92', '#C87923', '#596E7D'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': NAVY})

def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result

def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', path]).stdout)

def duration(path):
    result = probe(path)
    value = float(result['format'].get('duration', 0))
    if value <= 0:
        raise RuntimeError(f'No audio duration in {path}')
    return value

def make_background(scene, number, total):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor='white')
    fig.patches.extend([Rectangle((0, .982), 1, .018, transform=fig.transFigure, color=TEAL), Rectangle((.035, .025), .93, .122, transform=fig.transFigure, color='#EFF5F8')])
    fig.text(.035, .923, scene['title'], fontsize=25, fontweight='bold', va='center')
    fig.text(.965, .95, f'AI4S  |  {number:02}/{total:02}', ha='right', fontsize=10, color=GRAY)
    fig.text(.049, .102, textwrap.fill(scene['takeaway'], 100), fontsize=14, va='center', linespacing=1.4)
    fig.text(.035, .012, textwrap.fill(scene['source'], 150), fontsize=8.4, color=GRAY, va='bottom')
    fig.patches.append(Rectangle((.035, .155), .93 * number / total, .006, transform=fig.transFigure, color=TEAL))
    if scene['layout'] == 'portrait':
        for i, item in enumerate(scene['points']):
            fig.text(.49, .75 - i * .14, textwrap.fill(item, 36), fontsize=21, va='top', linespacing=1.4)
    elif scene['layout'] == 'csv':
        frame = pd.read_csv(ROOT / 'Research_and_Experiments/Evidence/2026-10-08_scorer_rebuild_job/ranking_fresh.csv')
        fig.text(.04, .855, 'Command to run the tool', fontsize=12, color=TEAL, fontweight='bold')
        fig.text(.04, .78, 'python inference.py --photos demo_photos\n    --out ranking.csv --sheet sheet.png', fontsize=13, family='DejaVu Sans Mono', va='center')
        fig.text(.76, .80, '12 scored\n5 manual reviews', fontsize=20, fontweight='bold', va='center', color=TEAL)
        data = []
        for _, row in frame.iterrows():
            data.append([str(int(row['rank'])) if pd.notna(row['rank']) else '', Path(row.photo).name, row['format'], 'Automatic' if row.route == 'automatic' else 'Manual review', f'{row.p_bad:.3f}' if pd.notna(row.p_bad) else 'No score'])
        ax = fig.add_axes([.035, .19, .93, .535]); ax.axis('off')
        table = ax.table(cellText=data, colLabels=['Rank', 'Photo', 'Format', 'Route', 'Review score'], cellLoc='left', colLoc='left', colWidths=[.075, .215, .23, .25, .23], bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(10.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#DCE5EA'); cell.set_linewidth(.6)
            if r == 0:
                cell.set_facecolor(NAVY); cell.set_text_props(color='white', weight='bold')
            elif r > 12:
                cell.set_facecolor('#FFF1DA')
            else:
                cell.set_facecolor('#F4F8FA' if r % 2 == 0 else 'white')
    elif scene['layout'] == 'reproduction':
        cards = [('F03', 'Rebuild completed', ['Photo pipeline', 'One nucleus split', 'About 42 minutes'], TEAL), ('F05', 'Normal path passed', ['Fresh installation', '17-photo CPU demo', 'Scorer versions match'], TEAL), ('F06', 'Extra strict test failed', ['Kaggle status: ERROR', 'No useful failure log', 'Not counted as passed'], AMBER)]
        for i, (tag, heading, body, color) in enumerate(cards):
            x = .035 + i * .316
            fig.patches.append(FancyBboxPatch((x, .29), .296, .50, boxstyle='round,pad=.006', transform=fig.transFigure, facecolor='#F3F7F9', edgecolor=color, linewidth=1.5))
            fig.text(x + .023, .72, tag, fontsize=30, color=color, fontweight='bold')
            fig.text(x + .023, .62, textwrap.fill(heading, 20), fontsize=18, fontweight='bold')
            fig.text(x + .023, .51, '\n\n'.join(body), fontsize=14, va='top')
        fig.text(.05, .205, 'This is a scoped reproduction record, not a rebuild of every study.', fontsize=17, color=GRAY)
    elif scene['layout'] == 'limits':
        for x, title, body, color in [(.04, 'Measured here', ['Better photo ranking', 'Separate nucleus results', 'Simulator control results'], TEAL), (.52, 'Still needs evidence', ['New chip laboratories', 'Human minutes saved', 'Real camera noise', 'Physical chip control'], AMBER)]:
            fig.patches.append(FancyBboxPatch((x, .27), .43, .55, boxstyle='round,pad=.006', transform=fig.transFigure, facecolor='#F3F7F9', edgecolor=color, linewidth=1.5))
            fig.text(x + .025, .74, title, fontsize=23, color=color, fontweight='bold')
            fig.text(x + .025, .64, '\n\n'.join(body), fontsize=20, va='top')
    elif scene['layout'] == 'credits':
        lines = ['Organ-on-a-Chip images: Movčana and colleagues', 'Zenodo 10203721 | Record: CC BY 4.0; paper: CC BY-SA', 'Nucleus data: BBBC039 / BBBC038, Broad Institute, CC0', 'Models and tools: DINOv2, U-Net, PyTorch, timm, scikit-learn', 'Control: published droplet equations; Stable-Baselines3', 'Narration: macOS Samantha synthetic voice | No music', 'Full source details and experiment record: Kaggle write-up']
        for i, line in enumerate(lines):
            fig.text(.055, .81 - i * .074, line, fontsize=16, va='center')
    out = FRAMES / f'{scene["id"]}_background.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def timestamp(seconds):
    ms = round(seconds * 1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-audio', action='store_true')
    args = parser.parse_args()
    story_path = HERE / 'storyboard.json'
    story = json.loads(story_path.read_text())
    verified = json.loads((HERE / 'script_verification.json').read_text())
    assert verified['script_sha256'] == hashlib.sha256(story_path.read_bytes()).hexdigest()
    assert verified['status'].startswith('PASS')
    scenes = story['scenes']
    backgrounds = [make_background(scene, i + 1, len(scenes)) for i, scene in enumerate(scenes)]
    if args.prepare_only:
        print(json.dumps({'backgrounds': len(backgrounds), 'script_verified': True})); return
    subtitles, schedule = [], []
    cursor = 0.0
    for scene, background in zip(scenes, backgrounds):
        pieces = []
        local_cursor = 0.0
        for n, sentence in enumerate(scene['sentences']):
            spoken = sentence.replace('AI4S', 'A I four S').replace('DINOv2', 'Dino vee two').replace('AUC', 'A U C').replace('B02', 'B zero two').replace('B03', 'B zero three')
            text_path = AUDIO / f'{scene["id"]}_{n:02}.txt'
            audio_path = AUDIO / f'{scene["id"]}_{n:02}.aiff'
            if args.reuse_audio and audio_path.exists():
                assert text_path.read_text() == spoken
            else:
                text_path.write_text(spoken)
                run(['say', '-v', story['voice'], '-r', str(story['speech_rate']), '-f', text_path, '-o', audio_path])
            seconds = duration(audio_path)
            pieces.append(audio_path)
            subtitles.append((cursor + local_cursor, cursor + local_cursor + seconds, sentence))
            local_cursor += seconds
        concat_file = AUDIO / f'{scene["id"]}_concat.txt'
        concat_file.write_text('\n'.join(f"file '{x}'" for x in pieces) + '\n')
        wave = AUDIO / f'{scene["id"]}.wav'
        run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file, '-ar', '24000', '-c:a', 'pcm_s16le', wave])
        audio_seconds = duration(wave)
        seconds = math.ceil(max(scene['minimum_seconds'], audio_seconds + .8) * 24) / 24
        schedule.append({'id': scene['id'], 'start': cursor, 'duration': seconds, 'audio_duration': audio_seconds, 'background': str(background), 'audio': str(wave), 'scene': scene})
        cursor += seconds
        print(json.dumps({'narration': scene['id'], 'seconds': seconds}), flush=True)
    if cursor >= story['maximum_seconds']:
        raise RuntimeError(f'Video would exceed the limit: {cursor:.2f} seconds')
    (HERE / 'schedule.json').write_text(json.dumps(schedule, indent=2))
    srt = '\n\n'.join(f'{i+1}\n{timestamp(start)} --> {timestamp(end)}\n{textwrap.fill(text, 76)}' for i, (start, end, text) in enumerate(subtitles)) + '\n'
    (HERE / 'ai4s_demo.srt').write_text(srt)
    (HERE / 'ai4s_demo.vtt').write_text('WEBVTT\n\n' + re.sub(r'(\d{2}:\d{2}:\d{2}),(\d{3})', r'\1.\2', srt))
    for entry in schedule:
        scene = entry['scene']; clip = CLIPS / f'{entry["id"]}.mp4'
        command = ['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '24', '-i', entry['background']]
        if scene.get('asset'):
            command += ['-loop', '1', '-framerate', '24', '-i', WRITEUP / scene['asset'], '-i', entry['audio']]
            x, y, w, h = (56, 144, 770, 736) if scene['layout'] == 'portrait' else (70, 144, 1780, 730)
            graph = f'[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=white[asset];[0:v][asset]overlay={x}:{y}:shortest=1,format=yuv420p[v]'
            command += ['-filter_complex', graph, '-map', '[v]', '-map', '2:a']
        else:
            command += ['-i', entry['audio'], '-map', '0:v', '-map', '1:a', '-vf', 'format=yuv420p']
        command += ['-t', f'{entry["duration"]:.6f}', '-r', '24', '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage', '-crf', '23', '-pix_fmt', 'yuv420p', '-threads', '2', '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11,apad', '-c:a', 'aac', '-b:a', '96k', '-ar', '48000', '-ac', '2', '-movflags', '+faststart', clip]
        run(command)
        print(json.dumps({'rendered': entry['id']}), flush=True)
    clip_list = HERE / 'concat_clips.txt'
    clip_list.write_text('\n'.join(f"file '{CLIPS / (entry['id'] + '.mp4')}'" for entry in schedule) + '\n')
    joined = HERE / 'ai4s_demo_no_subtitles.mp4'
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', clip_list, '-c', 'copy', '-movflags', '+faststart', joined])
    final = HERE / 'AI4S_Demo.mp4'
    run(['ffmpeg', '-v', 'error', '-y', '-i', joined, '-i', HERE / 'ai4s_demo.srt', '-map', '0:v', '-map', '0:a', '-map', '1:0', '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text', '-metadata:s:s:0', 'language=eng', '-metadata', 'title=AI4S Chip Photo Review', '-metadata', 'comment=Recorded verified outputs; synthetic narration; no new experiments', '-movflags', '+faststart', final])
    metadata = probe(final)
    seconds = float(metadata['format']['duration'])
    assert seconds < 300
    assert int(metadata['format']['size']) < 100_000_000
    streams = metadata['streams']
    assert any(s.get('codec_name') == 'h264' and s.get('width') == 1920 and s.get('height') == 1080 for s in streams)
    assert any(s.get('codec_name') == 'aac' for s in streams)
    assert any(s.get('codec_type') == 'subtitle' for s in streams)
    run(['ffmpeg', '-v', 'error', '-i', final, '-f', 'null', '-'])
    report = {'status': 'Encoded and fully decoded without ffmpeg errors; visual and audio-content review still required', 'duration_seconds': seconds, 'size_bytes': int(metadata['format']['size']), 'width': 1920, 'height': 1080, 'subtitle_entries': len(subtitles), 'audio': 'AAC, synthetic Samantha narration', 'video': 'H.264', 'under_five_minutes': True, 'under_kaggle_100mb': True, 'sha256': hashlib.sha256(final.read_bytes()).hexdigest(), 'script_sha256': verified['script_sha256']}
    (HERE / 'video_verification.json').write_text(json.dumps(report, indent=2))
    (HERE / 'ffprobe.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
