import re

with open('D:/code/test/camera-interact.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Add debug update code after FPS display
# The FPS block ends with: fpsEl.textContent = `FPS: ${STATE.fps}`;
# We need to add debug info OUTSIDE the if block, running every 15 frames

old = "    fpsEl.textContent = `FPS: ${STATE.fps}`;\n  }"
new = """    fpsEl.textContent = `FPS: ${STATE.fps}`;
  }

  // Debug panel (every 15 frames)
  if (STATE.frameCount % 15 === 0) {
    const dbg = document.getElementById('debug');
    if (dbg) {
      const hr = STATE.handResult ? (STATE.handResult.landmarks ? STATE.handResult.landmarks.length : 0) : 0;
      const fr = STATE.faceResult ? (STATE.faceResult.faceLandmarks ? STATE.faceResult.faceLandmarks.length : 0) : 0;
      const pr = STATE.poseResult ? (STATE.poseResult.landmarks ? STATE.poseResult.landmarks.length : 0) : 0;
      const vOk = video.videoWidth > 0 && !video.paused;
      dbg.textContent = 'vid:' + (vOk ? 'OK' : 'NO') + ' hands:' + hr + ' faces:' + fr + ' poses:' + pr + ' skip:' + STATE.handFrameSkip + '/' + STATE.faceFrameSkip + '/' + STATE.poseFrameSkip;
    }
  }"""

count = content.count(old)
print(f"Found {count} occurrences of FPS line")

if count == 1:
    content = content.replace(old, new)
    print("Replaced successfully")
else:
    print("ERROR: expected 1 match")
    # find it
    idx = content.find("fpsEl.textContent")
    print(repr(content[idx:idx+80]))

with open('D:/code/test/camera-interact.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
