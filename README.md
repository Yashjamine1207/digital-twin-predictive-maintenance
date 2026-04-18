# 🏭 AI-Driven Digital Twin for Predictive Maintenance

### A smart system that watches your machine in real time, predicts when it will break, and tells you exactly how to run it more efficiently — all automatically.

🔗 **Live Demo:** [Click here to open the dashboard](https://digital-twin-predictive-maintenance-r76mwvgtyvdnzzchghpcta.streamlit.app/)

---

## 🤔 What Problem Does This Solve?

Imagine you run a factory. Your machines run all day, every day.

- One day, a machine breaks down without warning
- Your entire production line stops
- Workers are standing idle
- You are losing thousands of pounds every hour
- An engineer arrives, diagnoses the problem, orders a part, and fixes it — 2 days later

**This is called reactive maintenance — you fix it after it breaks.**

This project solves that problem by predicting the breakdown BEFORE it happens, so you can fix it on your schedule, not the machine's schedule.

---

## 💡 The Simple Idea Behind This Project

Think of this system like a **doctor for your machine.**

- A doctor checks your blood pressure, temperature, and heartbeat to catch illness early
- This system checks the machine's temperature, speed, torque, and wear to catch failure early
- Just like a doctor gives you medicine before you get seriously ill, this system tells operators to adjust settings before the machine breaks

---

## 🎯 What Does This System Actually Do?

It does **three things simultaneously:**

- **👁️ Watches** — reads live sensor data from the machine every second
- **🔮 Predicts** — tells you which type of failure is coming and how likely it is
- **📋 Recommends** — tells you the exact speed and force settings to run the machine at maximum output without breaking it

---

## 🏗️ How Was It Built? (Simple Explanation)

### Step 1 — Collecting and Understanding the Data
- Used a real industrial dataset with **10,000 machine readings**
- Each reading captured: temperature, speed, torque (rotational force), and tool wear
- This is like 10,000 health checkups for the machine

### Step 2 — Making the Data Smarter
- The raw data only had 5 measurements per reading
- We calculated **33 extra features** from those 5 — things like:
  - How much heat is being generated (temperature difference)
  - How much mechanical power is being used (speed × force)
  - Whether the machine is getting hotter or cooler over time
- This is like a doctor not just checking your temperature but also calculating your body mass index, heart rate trend, and stress levels from basic readings

### Step 3 — Teaching the System to Recognise Failure
- Trained an **AI brain (Neural Network)** to recognise 5 different machine states:
  - ✅ Healthy — everything is fine
  - 🌡️ Heat Dissipation Failure — machine is overheating
  - ⚡ Power Failure — electrical load problem
  - 💪 Overstrain Failure — machine is being pushed too hard
  - 🔧 Tool Wear Failure — the cutting tool is worn out
- The hardest part: failures only happen 3.4% of the time, so the AI had a tendency to just say "healthy" for everything
- We fixed this by:
  - Artificially creating more failure examples (called SMOTE)
  - Punishing the AI 10× harder when it missed a real failure than when it gave a false alarm

### Step 4 — Finding the Best Settings Automatically
- Built a custom search engine (called **JAYA algorithm**) that tests thousands of combinations of speed and force settings
- It finds the combination that gives maximum output while keeping failure risk below 5%
- Think of it like a sat-nav that finds the fastest route while avoiding dangerous roads

### Step 5 — Building the Live Dashboard
- Wrapped everything into an **interactive website** anyone can use
- Operators can drag sliders to input current machine readings
- The system instantly shows: current health status, failure risk, and recommended settings

---

## 📊 Results — What Did the System Achieve?

| What We Measured | Result | What This Means |
|---|---|---|
| Accuracy at identifying all 5 states | **98.4%** | Almost never gets it wrong |
| Accuracy at catching real failures | **96.1%** | Catches 96 out of every 100 real faults |
| Response time | **18 milliseconds** | Faster than the blink of an eye |
| Production speed increase | **+14.5%** | More output from the same machine |
| Failure risk at recommended settings | **4.2%** | Well within the safe 5% limit |
| Energy waste reduction | **22%** | Less electricity wasted running inefficiently |
| Automatic safety when tool wears out | **100%** | System reduces speed automatically — no human needed |
| Estimated reduction in unplanned downtime | **85%** | Far fewer unexpected breakdowns |

---

## 💼 Why Does This Matter for a Business?

### The Cost of Doing Nothing
- A single unplanned machine breakdown in manufacturing can cost **£10,000–£100,000 per hour** in lost production
- Most factories still run on a "fix it when it breaks" approach
- Planned maintenance visits cost a fraction of emergency repairs

### What This System Gives a Business

- **💰 Saves money** — fewer emergency repairs, less wasted materials, less idle workforce
- **📈 Makes more money** — running at optimised settings produces 14.5% more output from the same machine
- **⚡ Saves energy** — 22% less energy wasted means lower electricity bills
- **🧑‍🔧 Helps engineers** — instead of guessing when to maintain, engineers get exact instructions
- **📱 No technical knowledge needed** — any operator can use the dashboard with simple sliders
- **🔄 Works continuously** — monitors the machine 24 hours a day, 7 days a week, without breaks

### Real World Example
> A factory runs 3 machines, 16 hours a day, 250 days a year.
> Each unplanned breakdown costs £15,000 and happens 6 times per year.
> **Total annual loss = £270,000**
>
> This system reduces breakdowns by 85%.
> **Annual saving = £229,500** — just from preventing downtime.
> Add the 14.5% throughput gain and energy savings on top.

---

## 🖥️ How to Use the Live Dashboard

1. **Open the link:** [Click here](https://digital-twin-predictive-maintenance-r76mwvgtyvdnzzchghpcta.streamlit.app/)
2. **Use the sliders** on the left side to enter the machine's current readings:
   - Rotational Speed (how fast the machine is spinning)
   - Torque (how much force it is applying)
   - Tool Wear (how long the tool has been in use)
   - Air and Process Temperature
3. **Watch the dashboard update instantly** — it shows:
   - Current machine health status
   - Probability of each type of failure
   - A safety gauge showing combined failure risk
   - 5 real-time alert indicators
4. **Click "Run JAYA Optimisation"** to get the recommended speed and torque settings
5. **The system automatically reduces speed** if the tool wear is above 200 minutes — no human intervention needed

---

## 🔬 Technologies Used

| Tool | What It Does |
|---|---|
| Python | The programming language everything is built in |
| TensorFlow | The AI framework that powers the failure prediction |
| JAYA Algorithm | Custom-built search engine for finding optimal settings |
| SMOTE | Technique for handling rare failure events in data |
| Streamlit | Framework for building the live interactive dashboard |
| Plotly | Creates the interactive charts and gauges |
| Scikit-learn | Data preparation and performance measurement |
| AI4I Dataset | Real industrial dataset with 10,000 machine readings |

---

## 📁 Project Files

| File | What It Is |
|---|---|
| `app.py` | The complete dashboard code |
| `digital_twin_model.keras` | The trained AI brain |
| `scaler.pkl` | Data normalisation settings |
| `feature_cols.pkl` | List of all 38 features the model uses |
| `requirements.txt` | List of software packages needed to run this |

---

## 👤 About This Project

- **Built by:** Yash Jamine
- **Type:** Self-directed individual project (portfolio piece)
- **Duration:** March 2026 – April 2026
- **Role:** Solo — handled everything from data engineering to cloud deployment
- **Context:** MSc Mechanical Engineering Design, University of Manchester

---

## 🔗 Connect

- **GitHub:** [github.com/Yashjamine1207](https://github.com/Yashjamine1207)
- **Live App:** [Open Dashboard](https://digital-twin-predictive-maintenance-r76mwvgtyvdnzzchghpcta.streamlit.app/)
