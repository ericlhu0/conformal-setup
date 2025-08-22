# Running Instructions
First start a web server
```
cd experiments
python3 -m http.server 8000
```

  Then in your browser, open: http://localhost:8000/web_interface/experiment_data_editor.html

# Data
## Input Context

- **Current action description**:  
  “You are grabbing the user's arm at the wrist to reposition them. The pressure of the grasp just increased, and you also moved their wrist joint.”

- **Current state**  
  - **Contact forces** (specified 1 = low, 5 = high):  
    - `{entire arm: 2, upper arm: 1, forearm: 1, wrist: 2}`
  - **Joint angles** (specified min, max degrees):  
    - For each arm joint in `{elbow, wrist}`: `165 deg`

- **Current comfort threshold**  
  - **Current comfort sensitivity**: dictionary maps possible comfort sensitivities (1–5, higher = more sensitive) to probabilities that sum to 1  
    - For each body part in `{entire arm, upper arm, forearm, wrist}`:  
      `{0, 0.1, 0.8, 0.1, 0}`
  - **Current comfortable joint range** (min and max, in degrees, increments of 15):  
    - **Min**: `{0: 0.6, 15: 0.3, 30: 0.1, 45: 0, ...}`  
    - **Max**: `{..., 120: 0, 135: 0.1, 150: 0.3, 165: 0.6}`

---

## Received Feedback

- **Verbal feedback**  
  - **Specificity**: “my arm hurts”  
  - **Intensity**: mid — “that’s a little uncomfortable”

- **Facial expression**  
  - **Modality**: text  
  - **Intensity**: mid — frowning face