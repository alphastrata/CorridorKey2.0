//! End-to-end tests of ck-gui, each against its own window on a private Xvfb display.
//! They skip without Xvfb and ffmpeg; those that key footage also need the release `ck` and a GPU.
//! Each test keys its own clip: outputs land beside it, and tests clean them up.

mod harness;

use harness::{Gui, ck, clip, gpus, wait};

fn row(g: &mut Gui, name: &str) -> String {
    g.find(name).into_iter().find(|e| !e.value.is_empty() && e.kind != "Text").map_or_else(String::new, |e| e.value)
}

fn int(v: Option<toml::Value>) -> Option<i64> {
    v.and_then(|v| v.as_integer())
}

#[test]
fn starts_empty() {
    let Some(mut g) = Gui::start(None, &[]) else { return };
    assert!(g.status().contains("drop clips"), "status: {}", g.status());
    assert_ne!(g.find("No clip selected"), vec![]);
}

#[test]
fn adds_clips_given_on_the_command_line() {
    let clips = [clip(320, 180, 12), clip(256, 144, 8)];
    let Some(mut g) = Gui::start(None, &clips) else { return };
    assert_eq!(g.status(), "Added 2 clips.");
    assert_eq!(row(&mut g, "pattern_320x180_12"), "Ready");
    assert_eq!(row(&mut g, "pattern_256x144_8"), "Ready");
    wait(5, "two saved clips", || (g.saved("queue.1").is_some()).then_some(()));
}

#[test]
fn selecting_and_removing_clips() {
    let clips = [clip(320, 180, 12), clip(256, 144, 8)];
    let Some(mut g) = Gui::start(None, &clips) else { return };
    g.activate("pattern_256x144_8", "Rectangle");
    wait(5, "the second clip selected", || (int(g.saved("selected")) == Some(1)).then_some(()));
    g.activate("Remove from queue", "Btn");
    wait(5, "one clip left", || g.saved("queue.1").is_none().then_some(()));
    assert_eq!(g.find("pattern_256x144_8"), vec![]);
    assert_eq!(row(&mut g, "pattern_320x180_12"), "Ready");
}

#[test]
fn preferences_open_and_close() {
    let Some(mut g) = Gui::start(None, &[]) else { return };
    let open = |g: &mut Gui| g.find("PREFERENCES").iter().any(|e| e.visible);
    assert!(!open(&mut g));
    g.keys(&["ctrl", ","]);
    assert!(open(&mut g), "Ctrl+, opens the sheet");
    g.keys(&["esc"]);
    assert!(!open(&mut g), "Esc closes it");
    g.activate("Preferences (Ctrl+,)", "Btn");
    assert!(open(&mut g), "the gear opens it");
    g.activate("Done", "Btn");
    assert!(!open(&mut g), "Done closes it");
}

#[test]
fn preference_shares_persist() {
    let Some(mut g) = Gui::start(None, &[]) else { return };
    g.keys(&["ctrl", ","]);
    g.set("VRAM", "Slider", "40");
    g.set("CPU cores", "Slider", "20");
    wait(5, "saved shares", || {
        (int(g.saved("prefs.share.vram")) == Some(40) && int(g.saved("prefs.share.cpu")) == Some(20)).then_some(())
    });
}

#[test]
fn a_clip_overrides_the_resource_defaults() {
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    assert_eq!(g.one("CPU cores", "Slider").enabled, Some(false), "defaults are read-only until overridden");
    g.activate("Override the Preferences defaults", "CheckBox");
    assert_eq!(g.one("CPU cores", "Slider").enabled, Some(true));
    g.set("CPU cores", "Slider", "100");
    wait(5, "the clip's own share", || (int(g.saved("queue.0.settings.share.cpu")) == Some(100)).then_some(()));
    assert_eq!(int(g.saved("prefs.share.cpu")), Some(75), "the default is untouched");
}

#[test]
fn a_clip_chooses_its_gpus() {
    if gpus() == 0 {
        return;
    }
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    g.activate("Choose GPUs for this clip", "CheckBox");
    let all: Vec<i64> = (0..gpus()).map(|i| i64::try_from(i).unwrap_or_default()).collect();
    let saved = |g: &Gui| g.saved("queue.0.settings.gpus").and_then(|v| v.as_array().map(|a| a.iter().filter_map(toml::Value::as_integer).collect::<Vec<_>>()));
    wait(5, "every enabled GPU picked", || (saved(&g) == Some(all.clone())).then_some(()));
    g.one("Choose GPUs for this clip", "CheckBox");
    let gpu0 = g.find("type:CheckBox").into_iter().find(|e| e.label.starts_with("GPU0") && e.enabled == Some(true)).expect("GPU0 box");
    g.activate(&gpu0.label, "CheckBox");
    let rest: Vec<i64> = all[1..].to_vec();
    wait(5, "GPU0 dropped", || {
        let s = saved(&g);
        (if rest.is_empty() { s.is_none() } else { s == Some(rest.clone()) }).then_some(())
    });
}

#[test]
fn an_impossible_job_is_flagged_and_refused() {
    if gpus() == 0 {
        return;
    }
    let Some(mut g) = Gui::start(Some("[prefs.share]\npriority = 50\nvram = 5\ncpu = 75\n"), &[clip(1920, 1080, 12)]) else {
        return;
    };
    let problem = wait(5, "a problem shown", || g.reveal("id:Problem::body", "Text").map(|(_, e)| e));
    assert!(problem.label.contains("Even one frame"), "problem: {}", problem.label);
    g.activate("Run selected", "Btn");
    assert!(g.status().starts_with("Not started"), "status: {}", g.status());
    assert!(g.log(3).contains("cannot run"));
    assert_eq!(g.saved("queue.0.status").and_then(|v| v.as_str().map(String::from)).as_deref(), Some("Ready"));
}

#[test]
fn manual_window_presets_follow_edits() {
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    g.choose("id:Inspector::memory", "Manual");
    assert_eq!(g.one("id:Inspector::memory", "ComboBox").value, "Manual");
    assert_eq!(g.one("id:Inspector::window", "ComboBox").value, "48 / 24");
    g.set("id:Inspector::chunk", "SpinBox", "30");
    assert_eq!(g.one("id:Inspector::window", "ComboBox").value, "Custom");
    wait(5, "the manual window saved", || (int(g.saved("queue.0.settings.window.0")) == Some(30)).then_some(()));
}

#[test]
fn outside_edits_of_the_state_file_are_adopted() {
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    wait(5, "a saved state", || g.saved("prefs.share.cpu").map(|_| ()));
    let text = std::fs::read_to_string(g.state_path()).expect("state");
    std::fs::write(g.state_path(), text.replace("cpu = 75", "cpu = 33")).expect("edit state");
    wait(5, "the edit adopted", || (g.one("CPU cores", "Slider").value == "33").then_some(()));
}

#[test]
fn snapshots_cover_the_window() {
    let Some(mut g) = Gui::start(None, &[]) else { return };
    let path = g.snap("window");
    let (w, h) = image::image_dimensions(&path).expect("snapshot");
    assert!(w >= 800 && h >= 600, "{w}x{h}");
}

#[test]
fn keys_a_clip_end_to_end() {
    if ck().is_none() || gpus() == 0 {
        return;
    }
    let Some(mut g) = Gui::start(None, &[clip(256, 144, 10)]) else { return };
    g.activate("Run selected", "Btn");
    wait(300, "the clip keyed", || match row(&mut g, "pattern_256x144_10").as_str() {
        "Done" => Some(()),
        "Failed" => panic!("keying failed:\n{}", g.log(40)),
        _ => None,
    });
    let out = g.saved("queue.0.output_dir").and_then(|v| v.as_str().map(std::path::PathBuf::from)).expect("output dir");
    let files = std::fs::read_dir(&out).map_or(0, Iterator::count);
    assert!(files > 0, "no outputs in {}", out.display());
    drop(std::fs::remove_dir_all(out));
}

#[test]
fn stopping_a_running_clip() {
    if ck().is_none() || gpus() == 0 {
        return;
    }
    let Some(mut g) = Gui::start(None, &[clip(256, 144, 480)]) else { return };
    g.activate("Run selected", "Btn");
    wait(60, "the clip running", || (row(&mut g, "pattern_256x144_480") == "Running").then_some(()));
    g.activate("Stop", "Btn");
    wait(30, "the clip stopped", || (row(&mut g, "pattern_256x144_480") == "Stopped").then_some(()));
    if let Some(out) = g.saved("queue.0.output_dir").and_then(|v| v.as_str().map(std::path::PathBuf::from)) {
        drop(std::fs::remove_dir_all(out));
    }
}

fn flag(v: Option<toml::Value>) -> Option<bool> {
    v.and_then(|v| v.as_bool())
}

#[test]
fn log_and_stats_toggle_by_key_and_button() {
    let Some(mut g) = Gui::start(None, &[]) else { return };
    let log = flag(g.saved("layout.log")).unwrap_or(true);
    g.keys(&["ctrl", "l"]);
    wait(5, "the log toggled by key", || (flag(g.saved("layout.log")) == Some(!log)).then_some(()));
    g.activate("Log (Ctrl+L)", "Toggle");
    wait(5, "the log toggled by button", || (flag(g.saved("layout.log")) == Some(log)).then_some(()));
    g.activate("Stats for nerds (Ctrl+Shift+S)", "Toggle");
    assert_eq!(g.one("Stats for nerds (Ctrl+Shift+S)", "Toggle").checked, Some(true));
    wait(5, "stats saved", || (flag(g.saved("layout.stats")) == Some(true)).then_some(()));
}

#[test]
fn encoder_and_format_choices_persist() {
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    assert_eq!(g.one("value:Half float", "ComboBox").enabled, Some(false), "EXR type waits for EXR");
    g.choose("value:x264 · CPU", "HEVC · NVENC on the GPU");
    wait(5, "NVENC saved", || (flag(g.saved("queue.0.settings.nvenc")) == Some(true)).then_some(()));
    g.choose("value:Preview video", "EXR sequence · HDR");
    assert_eq!(g.one("value:Half float", "ComboBox").enabled, Some(true));
    g.choose("value:Half float", "Full float");
    wait(5, "full-float EXR saved", || (flag(g.saved("queue.0.settings.exr")) == Some(true)).then_some(()));
}

#[test]
fn outputs_never_all_switch_off() {
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    ["Alpha matte", "Foreground", "Cutout (RGBA)", "Checker preview"].iter().for_each(|o| {
        if g.one(o, "CheckBox").checked == Some(true) {
            g.activate(o, "CheckBox");
        }
    });
    wait(5, "outputs saved", || g.saved("queue.0.settings.outputs").map(|_| ()));
    let outs: Vec<bool> = g.saved("queue.0.settings.outputs").and_then(|v| v.as_array().map(|a| a.iter().filter_map(toml::Value::as_bool).collect())).unwrap_or_default();
    assert_eq!(outs, [true, false, false, false], "the alpha matte stays on");
}

#[test]
fn model_scale_enables_the_refiner() {
    let Some(mut g) = Gui::start(None, &[clip(320, 180, 12)]) else { return };
    assert_eq!(g.one("Edge refiner (best at ¼ and below)", "CheckBox").enabled, Some(false));
    g.choose("value:Full resolution", "¼ · fastest");
    assert_eq!(g.one("Edge refiner (best at ¼ and below)", "CheckBox").enabled, Some(true));
    wait(5, "scale saved", || (int(g.saved("queue.0.settings.scale")) == Some(2)).then_some(()));
}

#[test]
fn gpu_preferences_persist() {
    if gpus() < 2 {
        return;
    }
    let Some(mut g) = Gui::start(None, &[]) else { return };
    g.keys(&["ctrl", ","]);
    let gpu1 = g.find("type:CheckBox").into_iter().find(|e| e.label.starts_with("GPU1") && e.visible).expect("GPU1 box");
    g.activate(&gpu1.label, "CheckBox");
    g.activate("One clip per GPU at a time", "CheckBox");
    wait(5, "GPU1 off and placement together", || {
        let off = g.saved("prefs.gpus_off").and_then(|v| v.as_array().map(|a| a.iter().filter_map(toml::Value::as_integer).collect::<Vec<_>>()));
        let placement = g.saved("prefs.placement").and_then(|v| v.as_str().map(String::from));
        (off == Some(vec![1]) && placement.as_deref() == Some("Together")).then_some(())
    });
}

#[test]
fn run_all_skips_only_the_impossible_clip() {
    if ck().is_none() || gpus() == 0 {
        return;
    }
    let state = "[[queue]]\npath = \"P\"\nsource = \"P\"\nframes = false\nhint = \"None\"\nstatus = \"Ready\"\nprogress = 0.0\ndetail = \"\"\n\n[queue.settings.share]\npriority = 50\nvram = 1\ncpu = 75\n";
    let big = clip(1920, 1080, 12);
    let state = state.replace('P', &big.display().to_string());
    let Some(mut g) = Gui::start(Some(&state), &[clip(256, 144, 11)]) else { return };
    assert_eq!(row(&mut g, "pattern_1920x1080_12"), "Ready");
    g.activate("Run all", "Btn");
    assert!(g.log(10).contains("pattern_1920x1080_12 cannot run"), "log:\n{}", g.log(10));
    wait(300, "the small clip keyed", || match row(&mut g, "pattern_256x144_11").as_str() {
        "Done" => Some(()),
        "Failed" => panic!("keying failed:\n{}", g.log(40)),
        _ => None,
    });
    assert_eq!(row(&mut g, "pattern_1920x1080_12"), "Ready");
    g.activate("pattern_256x144_11", "Rectangle");
    assert_eq!(g.one("Alpha", "Tab").enabled, Some(true), "outputs are viewable after the run");
    g.activate("Alpha", "Tab");
    if let Some(out) = g.saved("queue.1.output_dir").and_then(|v| v.as_str().map(std::path::PathBuf::from)) {
        drop(std::fs::remove_dir_all(out));
    }
}
