"""
Interactive two-channel sleep EEG viewer with manual artifact annotation.
v2_2 - unfiltered display, 'Scoring finished' button, .csv written next to the EDF file.
Requires the Qt backend: run '%matplotlib qt' in the console first.
"""

#%% Imports and user configuration

import os
import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.transforms import blended_transform_factory
from matplotlib.backends.qt_compat import QtWidgets, QtCore
from scipy.signal import butter, sosfiltfilt
from lspopt import spectrogram_lspopt
from joblib import Parallel, delayed

START_DIR = r"/run/media/blue/M.2_2TB/wnzp_oa/Wearanize+_OA/Wearanize+OA_PlugNPlay_v1.1/"    # folder the file browser opens in
# the annotation .csv is written next to the selected EDF file, no output folder to set

LOWCUT, HIGHCUT, FILT_ORDER = 0.3, 30.0, 2    # band-pass settings, currently unused (display is unfiltered)
SPEC_WIN_SEC, SPEC_FMIN, SPEC_FMAX = 10.0, 0.3, 30.2    # spectrogram window and displayed band
WINDOW_SEC, XTICK_STEP_SEC = 30.0, 5.0    # time-series window width and x-tick spacing
SCROLL_STEP_SEC = WINDOW_SEC    # time shift per vertical mouse-wheel notch (wheel down = forward)
YLIM_INIT, YLIM_MIN, YLIM_MAX, YLIM_STEP = 150.0, 20.0, 1000.0, 5.0    # amplitude half-range: start, limits, slider step
AMP_STEP_UV = 50.0    # amplitude change per Shift+wheel / horizontal-wheel notch / arrow key press
SPEC_CLIP_PCT = (1.0, 99.0)    # percentiles used for the spectrogram colour limits

FIG_SIZE, PLOT_FONT_PT = (15.0, 10.0), 11    # matplotlib window size (inch) and font size
UI_FONT_PT = 11    # font size of the Qt windows
INPUT_WIN_SIZE, TABLE_WIN_SIZE = (940, 900), (900, 520)    # (width, height) in pixels

ART_LABELS = list(range(1, 5))    # valid artifact codes
ART_RULES = {1: 'No data',
             2: 'High noise',
             3: 'Spiky',
             4: 'M-shaped'}    # EDIT: shown in the input window
ANN_COLUMNS = ['channel_name', 'start_sec', 'end_sec', 'artifact_1', 'artifact_2', 'duration']
ANN_INT_COLUMNS = ['start_sec', 'end_sec', 'artifact_1', 'artifact_2', 'duration']
MIN_DURATION_SEC = 1    # a selection shorter than this is rejected

N_CORES = max(1, os.cpu_count() - 2)
random_state = 33

if 'qt' not in plt.get_backend().lower():
    raise RuntimeError(f"Qt backend required, current backend is '{plt.get_backend()}'. Run '%matplotlib qt' in the console, then re-run this script.")

#%% Helper: scoring-rule and navigation text shown in the input window

def build_rules_text():
    lines = ['<b>Navigation</b>',
             f'&bull; <b>Wheel down</b> moves the time window <b>forward</b> by {SCROLL_STEP_SEC:.0f} s, wheel up moves it back.',
             '&bull; <b>Left / right arrow</b>: move the time window by one full window.',
             '&bull; <b>Bottom slider</b>: jump anywhere in the recording.',
             f'&bull; <b>Shift + wheel up</b> increases the amplitude range by {AMP_STEP_UV:.0f} µV, Shift + wheel down decreases it.',
             f'&bull; <b>Horizontal wheel</b> (thumb wheel) does the same: right = +{AMP_STEP_UV:.0f} µV, left = -{AMP_STEP_UV:.0f} µV.',
             f'&bull; <b>Up / down arrow</b> or the second slider: change the amplitude range by {AMP_STEP_UV:.0f} µV.',
             '',
             '<b>Scoring rules</b>',
             '&bull; <b>Left-click</b> on a time-series pane sets the <b>start</b> of an artifact (nearest full second).',
             '&bull; <b>Right-click</b> on the <b>same</b> pane sets the <b>end</b>; a label window then opens.',
             f'&bull; Minimum duration is {MIN_DURATION_SEC} s; shorter or reversed selections are rejected.',
             '&bull; <b>Escape</b> discards a pending selection; a second left-click just moves the start point.',
             '&bull; <b>artifact_1</b> is mandatory, <b>artifact_2</b> is optional (left empty = no second label).',
             '&bull; Rows can be edited (times, labels) or deleted in the annotation table.',
             '&bull; Press <b>Scoring finished</b> (lower right of the plot) to save the .csv and end the session.',
             '&nbsp;&nbsp;&nbsp;Closing the plot window does the same.',
             '',
             '<b>Artifact codes</b>']
    lines += [f'&bull; <b>{code}</b> &mdash; {text}' for code, text in ART_RULES.items()]
    return '<br>'.join(lines)

#%% Helper: graphical input window (EDF file and two channels)

def ask_inputs():
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle('EEG artifact scorer - input')
    dlg.setStyleSheet(f'font-size: {UI_FONT_PT}pt;')
    dlg.resize(*INPUT_WIN_SIZE)
    state = {'path': None, 'labels': []}

    edit_path = QtWidgets.QLineEdit()
    edit_path.setReadOnly(True)
    edit_path.setPlaceholderText('no file selected')
    button_browse = QtWidgets.QPushButton('Browse...')
    row_file = QtWidgets.QHBoxLayout()
    row_file.addWidget(edit_path)
    row_file.addWidget(button_browse)

    box_1, box_2 = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    box_1.setEnabled(False)
    box_2.setEnabled(False)
    label_out = QtWidgets.QLabel('-')
    label_out.setWordWrap(True)
    label_hint = QtWidgets.QLabel('')
    label_hint.setStyleSheet(f'color: #b00; font-size: {UI_FONT_PT}pt;')

    rules = QtWidgets.QLabel(build_rules_text())
    rules.setTextFormat(QtCore.Qt.TextFormat.RichText)
    rules.setWordWrap(True)
    rules.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    group_rules = QtWidgets.QGroupBox('How to score')
    layout_rules = QtWidgets.QVBoxLayout(group_rules)
    layout_rules.setContentsMargins(14, 14, 14, 14)
    layout_rules.addWidget(rules)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(False)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText('Start scoring')
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    form = QtWidgets.QFormLayout()
    form.setVerticalSpacing(12)
    form.addRow('EDF file:', row_file)
    form.addRow('Channel 1:', box_1)
    form.addRow('Channel 2:', box_2)
    form.addRow('Output .csv:', label_out)
    form.addRow(label_hint)

    layout = QtWidgets.QVBoxLayout(dlg)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(14)
    layout.addLayout(form)
    layout.addWidget(group_rules, stretch=1)
    layout.addWidget(buttons)

    def validate():
        ok = state['path'] is not None and box_1.currentText() != '' and box_1.currentText() != box_2.currentText()
        label_hint.setText('The two channels must be different.' if state['path'] is not None and box_1.currentText() == box_2.currentText() else '')
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def browse():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, 'Select an EDF file', START_DIR, 'EDF files (*.edf *.EDF);;All files (*)')
        if path == '':
            return
        if os.path.splitext(path)[1].lower() != '.edf':
            QtWidgets.QMessageBox.warning(dlg, 'Not an EDF file', f"'{os.path.basename(path)}' is not an .edf file.\nSelect a file with the .edf extension.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            with pyedflib.EdfReader(path) as f:    # header only, signal data is not read here
                labels = [lab.strip() for lab in f.getSignalLabels()]
        except Exception as error:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(dlg, 'Cannot read file', f"{os.path.basename(path)} could not be opened:\n{error}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if len(labels) < 2:
            QtWidgets.QMessageBox.warning(dlg, 'Too few channels', f"'{os.path.basename(path)}' contains {len(labels)} channel(s); at least 2 are needed.")
            return
        state['path'], state['labels'] = path, labels
        edit_path.setText(path)
        for box, default in ((box_1, 0), (box_2, 1)):
            box.blockSignals(True)
            box.clear()
            box.addItems(labels)
            box.setCurrentIndex(default)
            box.setEnabled(True)
            box.blockSignals(False)
        label_out.setText(build_csv_path(path))
        print(f"      {len(labels)} channel(s) found: {labels}")
        validate()

    button_browse.clicked.connect(browse)
    box_1.currentTextChanged.connect(lambda _: validate())
    box_2.currentTextChanged.connect(lambda _: validate())
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return state['path'], [box_1.currentText(), box_2.currentText()]

#%% Helper: output path of the annotation file (next to the EDF file)

def build_csv_path(edf_path):
    return os.path.join(os.path.dirname(edf_path), os.path.splitext(os.path.basename(edf_path))[0] + '_artifact_scores.csv')

#%% Helper: read two channels from an EDF file

def read_edf_channels(edf_path, channel_names):
    print(f"[2/5] Reading EDF: {os.path.basename(edf_path)}")
    with pyedflib.EdfReader(edf_path) as f:
        labels = [lab.strip() for lab in f.getSignalLabels()]
        idx = [labels.index(name) for name in channel_names]
        fs_all = [float(f.getSampleFrequency(i)) for i in idx]
        if not np.allclose(fs_all, fs_all[0]):
            raise ValueError(f"The two channels have different sampling rates: {fs_all}")
        sigs = np.vstack([f.readSignal(i).astype(np.float32) for i in idx])
    fs = fs_all[0]
    print(f"      channels {channel_names} | fs = {fs:g} Hz | {sigs.shape[1]} samples ({sigs.shape[1] / fs / 60:.1f} min)")
    return sigs, fs

#%% Helper: Butterworth band-pass filter (zero-phase) - kept, currently not applied

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    sos = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)

#%% Helper: multitaper (lspopt) spectrogram of all channels

def spectrogram_plot_calc(signals, samp_rate, win_sec=SPEC_WIN_SEC, fmin=SPEC_FMIN, fmax=SPEC_FMAX, n_cores=N_CORES):
    def _compute_spectrogram(signal1):
        freqs, times, Spec = spectrogram_lspopt(signal1, samp_rate, nperseg=int(win_sec * samp_rate), noverlap=0)
        keep = (freqs >= fmin) & (freqs <= fmax)
        return Spec[keep, :], freqs[keep], times
    results = Parallel(n_jobs=min(n_cores, signals.shape[0]))(delayed(_compute_spectrogram)(signals[i, :]) for i in range(signals.shape[0]))
    specs_list, freqs, times = zip(*results)
    specs_all = np.array(specs_list)
    specs_all = (specs_all - specs_all.min()) / (specs_all.max() - specs_all.min()) * 10
    return {'specs': specs_all.astype(np.float32), 'freqs': freqs[0].astype(np.float32), 'times': times[0].astype(np.float32)}

#%% Collect the inputs, load data, and compute spectrograms

print("[1/5] Waiting for the input window")
inputs = ask_inputs()
if inputs is None:
    raise SystemExit("Input window cancelled, nothing to do.")
edf_path, ch_names = inputs
ANN_CSV = build_csv_path(edf_path)

sigs_raw, fs = read_edf_channels(edf_path, ch_names)
n_samples = sigs_raw.shape[1]
dur_sec = n_samples / fs
t_axis = np.arange(n_samples, dtype=np.float32) / fs

print("[3/5] Band-pass filtering skipped, the raw signals are displayed")
# sigs_disp = butter_bandpass_filter(sigs_raw, LOWCUT, HIGHCUT, fs, order=FILT_ORDER)    # uncomment to display the filtered signals
sigs_disp = sigs_raw

print(f"[4/5] Computing spectrograms on the raw signals, {N_CORES} core(s)")
plot_data = spectrogram_plot_calc(sigs_raw, fs)
spec_db = 10 * np.log10(np.maximum(plot_data['specs'], 1e-10))
vmin, vmax = np.percentile(spec_db, SPEC_CLIP_PCT)
print(f"      spectrogram shape {spec_db.shape} | colour limits {vmin:.1f} to {vmax:.1f} dB")

#%% Build the interactive panel

print("[5/5] Building the interactive panel")
plt.rcParams['keymap.pan'] = []    # free the 'p' key, keep arrow keys for navigation

fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, gridspec_kw={'height_ratios': [1, 1.3, 1, 1.3], 'hspace': 0.45})
fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.14)
fig.canvas.manager.set_window_title(f"Artifact scorer - {os.path.basename(edf_path)}")

lines, win_patches = [], []
spec_t = plot_data['times']
spec_f = plot_data['freqs']
extent = [float(spec_t[0]) - SPEC_WIN_SEC / 2, float(spec_t[-1]) + SPEC_WIN_SEC / 2, float(spec_f[0]), float(spec_f[-1])]

for i, ch_name in enumerate(ch_names):
    ax_spec, ax_sig = axs[2 * i], axs[2 * i + 1]

    ax_spec.imshow(spec_db[i], aspect='auto', origin='lower', extent=extent, cmap='seismic', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax_spec.set_xlim(0, dur_sec)
    ax_spec.set_ylabel('Frequency (Hz)', fontsize=PLOT_FONT_PT)
    ax_spec.set_xlabel('Time (s)', fontsize=PLOT_FONT_PT)
    ax_spec.tick_params(axis='both', labelsize=PLOT_FONT_PT)
    if i == 0:
        ax_spec.set_title(f'Spectrogram of channel {ch_name} [colorbar: (high-power) red>>white>>blue (low-power)]', fontsize=PLOT_FONT_PT, fontweight='bold')
    else:
        ax_spec.set_title(f'Spectrogram of channel {ch_name}', fontsize=PLOT_FONT_PT, fontweight='bold')
    patch = plt.Rectangle((0, 0), WINDOW_SEC, 1, transform=blended_transform_factory(ax_spec.transData, ax_spec.transAxes), facecolor='none', edgecolor='lime', linewidth=1.5, zorder=5)
    ax_spec.add_patch(patch)
    win_patches.append(patch)

    ln, = ax_sig.plot([], [], linewidth=0.6, color='k')
    ax_sig.set_ylim(-YLIM_INIT, YLIM_INIT)
    ax_sig.set_ylabel('Amplitude (µV)', fontsize=PLOT_FONT_PT)
    ax_sig.set_xlabel('Time (s)', fontsize=PLOT_FONT_PT)
    ax_sig.tick_params(axis='both', labelsize=PLOT_FONT_PT)
    ax_sig.grid(True, which='both', linewidth=0.4, alpha=0.5)
    ax_sig.set_title(f'Raw (unfiltered) signal of channel {ch_name}', fontsize=PLOT_FONT_PT, fontweight='bold')
    lines.append(ln)

ax_time = fig.add_axes([0.16, 0.06, 0.66, 0.025])
ax_amp = fig.add_axes([0.16, 0.02, 0.66, 0.025])
ax_done = fig.add_axes([0.865, 0.02, 0.115, 0.045])
t_max = max(0.0, dur_sec - WINDOW_SEC)
s_time = Slider(ax_time, 'Window start (s)', 0.0, t_max if t_max > 0 else 1e-6, valinit=0.0, valstep=1.0, color='0.6')
s_amp = Slider(ax_amp, 'Amplitude (± µV)', YLIM_MIN, YLIM_MAX, valinit=YLIM_INIT, valstep=YLIM_STEP, color='0.6')
b_done = Button(ax_done, 'Scoring finished', color='0.85', hovercolor='0.70')
for widget in (s_time, s_amp):
    widget.label.set_fontsize(PLOT_FONT_PT)
    widget.valtext.set_fontsize(PLOT_FONT_PT)
b_done.label.set_fontsize(PLOT_FONT_PT)
b_done.label.set_fontweight('bold')

def update_window(t0):
    t0 = float(np.clip(t0, 0.0, t_max))
    t1 = t0 + WINDOW_SEC
    i0, i1 = int(round(t0 * fs)), min(n_samples, int(round(t1 * fs)) + 1)
    ticks = np.arange(t0, t1 + 0.5 * XTICK_STEP_SEC, XTICK_STEP_SEC)
    for i in range(2):
        lines[i].set_data(t_axis[i0:i1], sigs_disp[i, i0:i1])
        axs[2 * i + 1].set_xlim(t0, t1)
        axs[2 * i + 1].set_xticks(ticks)
        win_patches[i].set_x(t0)
    fig.canvas.draw_idle()

def on_amp(val):
    for i in range(2):
        axs[2 * i + 1].set_ylim(-val, val)
    fig.canvas.draw_idle()

def step_time(n_windows):
    s_time.set_val(float(np.clip(s_time.val + n_windows * WINDOW_SEC, 0.0, t_max)))

def step_amp(n_steps):
    s_amp.set_val(float(np.clip(s_amp.val + n_steps * AMP_STEP_UV, YLIM_MIN, YLIM_MAX)))

def on_key(event):
    if event.key in ('right', 'left'):    # time navigation
        step_time(1 if event.key == 'right' else -1)
    elif event.key in ('up', 'down'):    # amplitude range
        step_amp(1 if event.key == 'up' else -1)

def on_scroll(event):    # vertical wheel over any pane -> time navigation, wheel down = forward
    if event.inaxes not in list(axs):
        return
    s_time.set_val(float(np.clip(s_time.val - event.step * SCROLL_STEP_SEC, 0.0, t_max)))

def on_qt_wheel(qt_event):    # Shift + wheel or horizontal wheel -> amplitude range
    delta = qt_event.angleDelta()
    shifted = bool(qt_event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
    if delta.x() != 0 or (shifted and delta.y() != 0):
        step_amp((delta.x() if delta.x() != 0 else delta.y()) / 120.0)
        qt_event.accept()
        return
    canvas_wheel_event(qt_event)    # fall through to matplotlib's own scroll_event

canvas_wheel_event = fig.canvas.wheelEvent
fig.canvas.wheelEvent = on_qt_wheel

s_time.on_changed(update_window)
s_amp.on_changed(on_amp)
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('scroll_event', on_scroll)
update_window(0.0)

#%% Annotation state, dataframe, and drawing helpers

SIG_AX = {1: 0, 3: 1}    # figure-axes index of a time-series pane -> channel index

ann_df = pd.DataFrame({'channel_name': pd.Series(dtype='object'), 'start_sec': pd.Series(dtype='Int32'), 'end_sec': pd.Series(dtype='Int32'), 'artifact_1': pd.Series(dtype='Int32'), 'artifact_2': pd.Series(dtype='Int32'), 'duration': pd.Series(dtype='Int32')})
ann_spans = {}    # row id -> [patch on the time-series pane, patch on the spectrogram pane]
ann_state = {'start_sec': None, 'ax_idx': None, 'next_id': 0, 'marker': None}    # pending selection, not yet in ann_df
session = {'finished': False}    # guards against saving twice

def draw_span(rid):
    remove_span(rid)
    ch_idx = int(ann_df.at[rid, 'channel_name'] == ch_names[1])
    t0, t1 = int(ann_df.at[rid, 'start_sec']), int(ann_df.at[rid, 'end_sec'])
    p_sig = axs[2 * ch_idx + 1].axvspan(t0, t1, facecolor='red', alpha=0.20, zorder=0)
    p_spec = axs[2 * ch_idx].axvspan(t0, t1, facecolor='none', edgecolor='red', linewidth=1.2, zorder=4)
    ann_spans[rid] = [p_sig, p_spec]

def remove_span(rid):
    for patch in ann_spans.pop(rid, []):
        patch.remove()

def clear_pending():
    if ann_state['marker'] is not None:
        ann_state['marker'].remove()
        ann_state['marker'] = None
    ann_state['start_sec'], ann_state['ax_idx'] = None, None

def parse_label(text):    # returns ('ok', value) | ('empty', None) | ('bad', None)
    text = text.strip()
    if text == '':
        return 'empty', None
    try:
        value = int(round(float(text)))
    except ValueError:
        return 'bad', None
    return ('ok', value) if value in ART_LABELS else ('bad', None)

#%% Live, editable dataframe viewer

ROLE_ID = QtCore.Qt.ItemDataRole.UserRole
tbl_win = QtWidgets.QWidget()
tbl_win.setWindowTitle(f"Artifact annotations - {os.path.basename(edf_path)}")
tbl_win.setStyleSheet(f'font-size: {UI_FONT_PT}pt;')
tbl_win.resize(*TABLE_WIN_SIZE)
tbl = QtWidgets.QTableWidget(0, 7)
tbl.setHorizontalHeaderLabels(ANN_COLUMNS + ['delete'])
tbl.horizontalHeader().setStretchLastSection(True)
tbl.verticalHeader().setDefaultSectionSize(int(UI_FONT_PT * 3.2))
QtWidgets.QVBoxLayout(tbl_win).addWidget(tbl)
tbl_win.show()

def on_label_change(rid, col, text):
    if rid not in ann_df.index:
        return
    if col == 'artifact_1' and text == '':    # artifact_1 is mandatory, ignore a blank selection
        return
    ann_df.at[rid, col] = pd.NA if text == '' else np.int32(text)

def on_row_delete(rid):
    if rid in ann_df.index:
        remove_span(rid)
        ann_df.drop(index=rid, inplace=True)
        print(f"row {rid} deleted")
        refresh_table()
        fig.canvas.draw_idle()

def on_time_edit(item):
    col_idx = item.column()
    if col_idx not in (1, 2):
        return
    rid, col = item.data(ROLE_ID), ANN_COLUMNS[col_idx]
    if rid not in ann_df.index:
        return
    try:
        new_val = int(round(float(item.text())))
    except ValueError:
        print(f"WARNING: '{item.text()}' is not a number; edit reverted")
        refresh_table()
        return
    other = int(ann_df.at[rid, 'end_sec' if col == 'start_sec' else 'start_sec'])
    start_sec, end_sec = (new_val, other) if col == 'start_sec' else (other, new_val)
    if not 0 <= new_val <= dur_sec:
        print(f"WARNING: {new_val} s is outside the recording (0-{dur_sec:.0f} s); edit reverted")
    elif end_sec - start_sec < MIN_DURATION_SEC:
        print(f"WARNING: selected duration is less than {MIN_DURATION_SEC} sec; edit reverted")
    else:
        ann_df.loc[rid, [col, 'duration']] = [np.int32(new_val), np.int32(end_sec - start_sec)]
        draw_span(rid)
        fig.canvas.draw_idle()
    refresh_table()

def refresh_table():
    tbl.blockSignals(True)
    tbl.setRowCount(0)
    for r, (rid, row) in enumerate(ann_df.iterrows()):
        tbl.insertRow(r)
        for c, col in enumerate(ANN_COLUMNS):
            if col in ('artifact_1', 'artifact_2'):
                box = QtWidgets.QComboBox()
                box.addItems([str(a) for a in ART_LABELS] if col == 'artifact_1' else [''] + [str(a) for a in ART_LABELS])
                box.setCurrentText('' if pd.isna(row[col]) else str(int(row[col])))
                box.currentTextChanged.connect(lambda text, rid=rid, col=col: on_label_change(rid, col, text))
                tbl.setCellWidget(r, c, box)
            else:
                item = QtWidgets.QTableWidgetItem('' if pd.isna(row[col]) else str(row[col]))
                item.setData(ROLE_ID, int(rid))
                if col not in ('start_sec', 'end_sec'):
                    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                tbl.setItem(r, c, item)
        button = QtWidgets.QPushButton('Delete')
        button.clicked.connect(lambda _=False, rid=rid: on_row_delete(rid))
        tbl.setCellWidget(r, 6, button)
    tbl.blockSignals(False)

tbl.itemChanged.connect(on_time_edit)
refresh_table()

#%% Helper: one modal window asking for both artifact labels

def ask_labels(ch_name, start_sec, end_sec):
    header = f"{ch_name}   |   {start_sec}-{end_sec} s   |   duration {end_sec - start_sec} s"
    dlg = QtWidgets.QDialog(fig.canvas.manager.window)
    dlg.setWindowTitle('Artifact labels')
    dlg.setStyleSheet(f'font-size: {UI_FONT_PT}pt;')
    edit_1, edit_2 = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    form = QtWidgets.QFormLayout(dlg)
    form.setVerticalSpacing(12)
    form.addRow(QtWidgets.QLabel(header))
    form.addRow(f"artifact_1, one of {ART_LABELS}:", edit_1)
    form.addRow(f"artifact_2, one of {ART_LABELS} (optional):", edit_2)
    form.addRow(buttons)
    while True:
        edit_1.setFocus()
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        status_1, art_1 = parse_label(edit_1.text())
        status_2, art_2 = parse_label(edit_2.text())
        if status_1 == 'ok' and status_2 in ('ok', 'empty'):
            return art_1, art_2
        print(f"WARNING: invalid label entry ('{edit_1.text()}', '{edit_2.text()}'); asking again")
        QtWidgets.QMessageBox.warning(dlg, 'Invalid label', f"artifact_1 must be one of {ART_LABELS}.\nartifact_2 must be one of {ART_LABELS} or left empty.")

#%% Mouse-driven annotation (LMB = start, RMB = end, Escape = discard)

def on_annotate_click(event):
    toolbar = getattr(fig.canvas, 'toolbar', None)
    if toolbar is not None and getattr(toolbar, 'mode', ''):    # a zoom/pan tool is active
        return
    if event.inaxes is None or event.xdata is None or event.button not in (1, 3):
        return
    ax_list = list(axs)
    ax_idx = ax_list.index(event.inaxes) if event.inaxes in ax_list else None
    if ax_idx not in SIG_AX:    # clicks on the spectrograms, sliders, and button are ignored
        return
    ch_idx = SIG_AX[ax_idx]
    t_click = int(round(float(np.clip(event.xdata, 0.0, dur_sec))))

    if event.button == 1:
        if ann_state['start_sec'] is not None:
            if ann_state['ax_idx'] == ax_idx:
                print(f"WARNING: no end point was set; start_sec reset to {t_click} s")
            else:
                print(f"WARNING: start and end points must be in the same graph; pending selection on {ch_names[SIG_AX[ann_state['ax_idx']]]} discarded")
            clear_pending()
        ann_state['start_sec'], ann_state['ax_idx'] = t_click, ax_idx
        ann_state['marker'] = event.inaxes.axvline(t_click, color='red', linestyle='--', linewidth=1.2, zorder=6)
        print(f"start_sec = {t_click} s on {ch_names[ch_idx]}")

    elif event.button == 3:
        start_sec = ann_state['start_sec']
        if start_sec is None:
            print("WARNING: right-click ignored, no start point has been set")
            return
        if ann_state['ax_idx'] != ax_idx:
            print(f"WARNING: end point clicked on {ch_names[ch_idx]} but the start point is on {ch_names[SIG_AX[ann_state['ax_idx']]]}; selection discarded")
            clear_pending()
        elif t_click - start_sec < MIN_DURATION_SEC:
            print(f"Selected duration is less than {MIN_DURATION_SEC} sec")
            clear_pending()
        else:
            labels = ask_labels(ch_names[ch_idx], start_sec, t_click)
            if labels is None:
                print("WARNING: label entry cancelled; selection discarded")
                clear_pending()
            else:
                art_1, art_2 = labels
                rid = ann_state['next_id']
                ann_state['next_id'] += 1
                ann_df.loc[rid, ANN_COLUMNS] = [ch_names[ch_idx], np.int32(start_sec), np.int32(t_click), np.int32(art_1), pd.NA if art_2 is None else np.int32(art_2), np.int32(t_click - start_sec)]
                clear_pending()
                draw_span(rid)
                print(f"row {rid}: {ch_names[ch_idx]} | {start_sec}-{t_click} s | duration {t_click - start_sec} s | labels {art_1}, {'-' if art_2 is None else art_2}")
    refresh_table()
    fig.canvas.draw_idle()

def on_annotate_key(event):
    if event.key == 'escape' and ann_state['start_sec'] is not None:
        print("Pending selection discarded (Escape)")
        clear_pending()
        fig.canvas.draw_idle()

#%% Finish the session: sort, save, and close both windows

def finish_session(event=None):
    global ann_df
    if session['finished']:
        return
    session['finished'] = True
    clear_pending()
    ann_df = ann_df.sort_values(['channel_name', 'start_sec']).reset_index(drop=True)
    ann_df[ANN_INT_COLUMNS] = ann_df[ANN_INT_COLUMNS].astype('Int32')
    print(f"\nScoring finished. Final annotation table ({len(ann_df)} row(s)):")
    print(ann_df.to_string() if len(ann_df) else "      (empty)")
    try:
        ann_df.to_csv(ANN_CSV, index=False)
        print(f"      saved to {ANN_CSV}")
    except OSError as error:
        print(f"ERROR: could not write {ANN_CSV}: {error}")
        print("      the table is still available in the variable 'ann_df'")
    tbl_win.close()
    plt.close(fig)

b_done.on_clicked(finish_session)
fig.canvas.mpl_connect('button_press_event', on_annotate_click)
fig.canvas.mpl_connect('key_press_event', on_annotate_key)
fig.canvas.mpl_connect('close_event', finish_session)

print("      ready: LMB = start of artifact, RMB = end, then enter the labels; Escape = discard pending selection")
plt.show()