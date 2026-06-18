# Test interview — exercise every feature

Open this on a second screen and read the **bold answers** aloud to the camera.
Each question is engineered to trigger specific parts of the report. Follow the
*(italic actions)* — they test emotions, eye contact, and speaking robustness.

## Start
```bash
source .venv/bin/activate
python app.py interview
```
1. (For English) press **`l`** until the bottom-left shows **`lang: English`** — or run `python app.py interview --lang en`.
2. Click **Start Recording**.
3. During the 2.5 s **calibration**, hold a **neutral, relaxed face** looking normally at the camera/screen.
4. Then answer the questions below. When done, press **q** (or **Stop**) → the report opens.

---

## Q1 — "Tell me about your background."
> **"My background is in software engineering. I studied computer science at university and I have about three years of experience. I started as a backend developer at a startup, where I worked mostly with Python and SQL databases, and later I moved into machine learning."**

Tests: **background** (semantic — note you never say the word "background" as a label), hard skills, key concepts.

## Q2 — "What technologies do you work with?"
> **"On the backend I use Python with Django and Flask, and a lot of SQL for Postgres. For machine learning I train models with TensorFlow and scikit-learn. I deploy on AWS using Docker and Kubernetes, and on the frontend I've built dashboards with React. I also use Git and write unit tests."**

Tests: lots of **hard skills** (Programming, Databases, Data/ML, Cloud/DevOps, Web/Frontend, Testing), **key concepts**.

## Q3 — "Tell me about a time you worked in a team."
> **"In my last project I led a small team of three developers. Communication was important, so I organized daily check-ins and kept everyone aligned. When we hit a difficult bug, I collaborated closely with my colleagues to solve it. My strengths are teamwork, problem-solving, and staying organized under deadlines."**

Tests: **soft skills** (Leadership, Communication, Teamwork, Problem-solving, Time management), impact verbs.

## Q4 — "What are you most proud of?"  *(emotion test)*
*Say this while genuinely **smiling**:*
> **"I'm really proud and excited about a recommendation system I built — it improved engagement by forty percent, and I was so happy when it launched."**

*Then **act surprised** — raise your eyebrows and widen your eyes — and say:*
> **"And honestly, I was surprised it worked so well on the first try!"**

Tests: **happy** (smile), **surprise** (eyes + brows). Confirms the fix: surprise needs eyes/brows, not just an open mouth.

## Q5 — "Walk me through how you solve a difficult problem."  *(speaking-robustness test)*
*Talk **continuously and normally** for ~30 s with a relaxed face — lots of mouth movement, no exaggerated expressions.*
> **"Usually I start by breaking the problem into smaller parts, I reproduce it, I check the logs, I form a hypothesis, and then I test it step by step until I find the root cause."** *(keep going naturally)*

Tests: should stay **neutral** — talking should NOT flip to "surprise."

## Q6 — "Where do you see yourself in five years?"  *(eye-contact test)*
*While answering, deliberately **look away** from the camera several times — glance to the side, down at your phone — then look back.*
> **"I'd like to grow into a senior or lead role, keep learning, and mentor junior developers."**

Tests: **eye contact %** should clearly **drop** (not stay ~90%).

## Q7 — "Tell me about a weakness."  *(fillers + pace test)*
*Read this **slowly**, leaving the "um/uh/like" in:*
> **"Um... so, like, I guess, uh... I sometimes, you know, basically spend a bit too much time on, um, small details..."**

*Then pause **silently for ~2 seconds**, then finish:*
> **"...but I've been working on it."**

Tests: high **filler rate**, **slow pace**, and a **real pause** (the % should reflect that one pause, not be inflated).

## Q8 — "Spune-mi despre tine, în română."  *(Romanian + diacritics test)*
> **"Am studiat informatică și am aproximativ trei ani de experiență. Am lucrat ca dezvoltator backend cu Python și baze de date SQL, iar mai târziu am construit un model de machine learning. Îmi place foarte mult să lucrez în echipă și să rezolv probleme dificile."**

Tests: **Romanian transcription** + **diacritics** (ă â î ș ț render correctly), hard skills (Python, SQL, machine learning), soft skills (teamwork, problem-solving), background — all detected in Romanian.

---

## After you press `q` — verify in the report

- [ ] **Hard skills mentioned**: Programming, Databases, Data/ML, Cloud/DevOps, Web/Frontend, Testing
- [ ] **Soft skills mentioned**: Communication, Teamwork, Leadership, Problem-solving, Time management
- [ ] **Background** section / "Background >" shows your experience sentences
- [ ] **Key concepts**: precise phrases (e.g. "machine learning", "recommendation system", "unit tests")
- [ ] **Eye contact %** is low-ish (because of Q6), not ~90%
- [ ] **Pauses %** is small (you only really paused once)
- [ ] **Filler words** count is high (because of Q7)
- [ ] **Peak emotional moments**: a happy peak around Q4 and a surprise peak at the "surprised" line — **each shows the exact line you said** at that moment
- [ ] **Interview integrity (heuristic)**: now counts **discrete look-away events** (head pose + gaze), so even a **1–2 second glance** registers. Glance **down at your phone**, then **up at a monitor**, then **to a side screen** → it should list *"3 look-away events [1x down (phone/notes); 1x up (monitor above); 1x to the right (side screen)]"* and raise the score. A run looking at your screen the whole time → score ~0. *(Second-person flag: have someone briefly step into frame.)* Tip: look at your screen **normally** during the 2.5 s calibration — that sets your personal baseline.
- [ ] **Full transcript >**: Romanian shows correct diacritics, not `??`
- [ ] **Click `more >`** on a hard skill, a soft skill, "Background", and "Full transcript" → each detail window shows the **exact lines you said** with timestamps
- [ ] Click a **soft-skill score** (left bars) → shows representative lines + a "scored from voice & face delivery" note

## Then test gestures (separate)
```bash
python app.py gestures --dry-run     # safe: logs actions, controls nothing
```
- [ ] ☝️ **point your index finger** → the cursor follows it; **hold it still ~2.5 s** (after moving) → it left-clicks (a green ring fills to show progress). Just holding the pose without moving should NOT click.
- [ ] ✊ open→fist (close window — recognizes the fist from any angle now), ☝️ **draw a circle** with your index finger (volume up/down), 👍/👎 (play-pause / mute), ✌️ peace (screenshot), 🤏 pinch — each appears in the **Action Log**
- [ ] If the finger-circle changes volume the *wrong* way, just rotate the other direction; tell me and I'll flip the default. (Cursor + click need Terminal in **Accessibility**.)
- [ ] After one gesture, wave your hand around → you see **WAIT** and no random gestures fire
- [ ] Then `python app.py gestures` (live) → ✌️ peace actually saves a screenshot to your Desktop
