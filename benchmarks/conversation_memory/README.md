# Conversation-memory regression corpus

This versioned synthetic corpus tests whether an explicitly stated fact from an
older completed turn remains available for a paraphrased follow-up after fifteen
unrelated turns. It compares full raw history, the previous recent-only policy,
and the production summary plus embedding-selection policy.

The dataset was inspected while Point 4 was implemented. It is not held out,
does not contain production traffic, and does not measure model-generated answer
accuracy. The exact outcome is deterministic fact-retention/task accuracy.
