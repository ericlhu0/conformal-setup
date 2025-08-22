in playground/ i have files to do experiments with data, all specified with jsons. 
  this is my experiment: I work in assistive robotics, which is using a robot to help
   people with disabilities do daily tasks. For some of these tasks, physical contact
   with the user is necessary. However, current systems can’t safely interpret 
  freeform user feedback (voice, facial expression, etc). Some methods are restricted
   to structured feedback (specify a body part, and if you want more or less force), 
  and others that incorporate LLMs don’t ensure that the interpretation of feedback 
  is safe, which suffers from user feedback being often ambiguous or contradictory, 
  and LLMs being overconfident.

  I’m starting a research project on allowing LLMs/VLMs to safely interpret human 
  feedback. Currently, I want to see how current LLMs perform when interpreting 
  feedback, particularly when there are contradictions and ambiguities.

  The specific tasks I’m thinking about involve the robot manipulating the human’s 
  arm, and the human giving some type of feedback about their comfort. Maybe they’re 
  uncomfortable with the amount of pressure put on their arm, or with the position 
  their arm is in. They would give some type of feedback, and the robot should adjust
   its internal model of what the human is comfortable with


these are some research questions i am hoping to answer
how unconfident is the model when there is disagreement between speech and facial expression?
how unconfident is the model when there is source of discomfort specificity ambiguity in the feedback?
does the model correctly adapt to different intensities of discomfort feedback?
how does the modality of the facial expression (image vs text representation) affect model outputs?
what is the difference when reading a model’s uncertainty from its single-token output distribution vs outputted verbalized uncertainty?
