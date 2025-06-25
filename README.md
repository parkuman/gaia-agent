---
title: parker's GAIA Agent - HF Agents Course
emoji: 🧠
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# HF Agents Course - GAIA Agent

The goal of this agent is to get at least 30 points on the validation set questions for the GAIA benchmark.

## Fetching and submitting questions/answers

[Huggingface Swagger](https://agents-course-unit4-scoring.hf.space/docs)

GAIA prompt for nice answers, though we should remove `FINAL ANSWER:`

```
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
```

## Leaderboard

[Student Leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard)

## TODO

- [ ] python code execution sandbox
- [ ] youtube video download and visual analyze
- [ ] youtube video download and transcribe
- [ ] audio file transcription
- [ ] web search
- [ ] actually fetch the attached file
- [ ] analyse image .. ?
- [ ] excel file reader
