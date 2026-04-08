# SIGDM_my

<p align="center">
  <h1 align="center">SIGDM_my</h1>
  <p align="center"><b>Diffusion Motion, Unlocked.</b></p>
  <p align="center">
    A creative research codebase for generating human motion from <b>text</b>, <b>actions</b>, and <b>audio</b>.
  </p>
</p>

---

## What Is SIGDM_my?

SIGDM_my is a diffusion-powered motion generation project built for people who want to explore controllable, expressive, and multi-condition human motion synthesis.

Instead of focusing on one narrow setup, this repository brings several motion-generation directions into one place:

- Text-to-motion generation
- Action-to-motion generation
- Audio/speech-driven gesture generation

It is designed for rapid experimentation, clean modular hacking, and practical iteration.

---

## Core Idea

At its heart, SIGDM_my treats motion as a rich temporal signal that can be guided by different conditions.

- **Language** can describe intent and style.
- **Action labels** can constrain motion categories.
- **Audio signals** can drive rhythm and gesture flow.

The goal is not just to generate motion, but to make generation controllable, extensible, and fun to build on.

---

## Project Personality

This project is built with a clear philosophy:

- **Research-friendly**: easy to inspect, modify, and extend.
- **Modular-by-default**: data, model, diffusion, evaluation, and sampling are separated cleanly.
- **Multi-branch creativity**: includes a dedicated audio2pose direction under the same ecosystem.
- **Practical over decorative**: scripts and structure are made to run real experiments, not just look good.

---

## What's Inside

SIGDM_my includes:

- Motion data loaders for multiple datasets
- Diffusion process implementations
- MDM-style model components
- Sampling/editing entry points
- Training loops and utilities
- Evaluation modules for different motion settings
- Audio-to-gesture pipeline components

This gives you an end-to-end playground for motion generation research and prototyping.

---

## Who This Repo Is For

SIGDM_my is a strong fit if you are:

- Building controllable motion generation systems
- Exploring diffusion models for temporal human dynamics
- Prototyping text/action/audio-conditioned animation pipelines
- Extending existing motion-generation methods with custom modules

---

## Design Direction

SIGDM_my moves toward a unified motion intelligence stack:

- One codebase, multiple condition modalities
- Flexible architecture for new controls and datasets
- Better reproducibility through structured project organization
- Fast idea-to-implementation cycles

---

## Closing

SIGDM_my is not just a model dump. It is a motion-generation workshop:

**open, hackable, and built for iteration.**

If you want to create the next version of controllable human motion generation, this is your launchpad.
