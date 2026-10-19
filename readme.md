# Stress Reduction with Music — Study Repository

This repository contains the experimental test code and data analysis scripts developed for the research study **"Stress Reduction with Music"**.

## Project Overview

This study investigated whether listening to music modulates physiological stress responses during acute stress induction and subsequent recovery[cite: 1]. Two within-subject experiments utilized a computerized Stroop task to induce cognitive stress, monitoring participants' physiological responses through Heart Rate Variability (HRV via RMSSD) and subjective stress ratings[cite: 1, 2]. 

## Project Structure

The codebase is organized into modular scripts for data processing and experiment automation:

* **Experiment Automation:** Python scripts used to automate experimental procedures, including the presentation of the computerized Stroop task and collection of real-time subjective stress ratings[cite: 2].
* **Data Import & Preprocessing:** R scripts utilizing `dplyr` to iterate through raw participant directories, parsing physiological summary files (`*_physio_summary.csv`) and questionnaire responses (`*_questionnaires.csv`)[cite: 2].
* **Statistical Analysis:** Scripts implementing repeated-measures ANOVAs, Mauchly's test of sphericity, and non-parametric paired Wilcoxon tests to evaluate RMSSD data across conditions[cite: 2, 3].
* **Visualisation:** R scripts using `ggplot2` to generate comparative boxplots (with jittered individual observations) and individual trajectory plots for both experiments[cite: 2, 3, 4].

## Repository Contents

All files included in this repository represent the original code tests and analysis scripts used to process the unnormalized dataset for this study[cite: 1, 2].