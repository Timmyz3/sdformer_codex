Title: A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection

URL Source: http://ieeexplore.ieee.org/document/11509564

Published Time: Sat, 27 Jun 2026 06:22:33 GMT

Markdown Content:
*   [Download PDF](https://ieeexplore.ieee.org/document/ "You do not have access to this PDF")
*    Download References 
*   [Request Permissions](https://ieeexplore.ieee.org/document/ "Request permission for reuse.")
*   [Save to](https://ieeexplore.ieee.org/document/)
*   [Alerts](https://ieeexplore.ieee.org/document/)

## Abstract:

Optical flow estimation with event-based cameras could capture per-pixel motion field efficiently for intelligent scenarios. However, numerous redundant operations and la...[Show More](https://ieeexplore.ieee.org/document/)

## Metadata

## Abstract:

Optical flow estimation with event-based cameras could capture per-pixel motion field efficiently for intelligent scenarios. However, numerous redundant operations and large unnecessary external-memory-access (EMA) hinder its expected high-efficiency. To alleviate this, we demonstrate an accelerator with redundancy speculation, bit-width-aware compression and similarity detection, decreasing overall operations and EMA to 0.20x and 0.08x, while reducing corresponding energy and latency cost to 0.12x and 0.19x, respectively. In evaluation, an EMA-included energy efficiency of 14.07 TOPS/W is achieved.

**Date of Conference:** 19-23 April 2026

**Date Added to IEEE _Xplore_:** 13 May 2026

**ISBN Information:**

## ISSN Information:

**Conference Location:** Seattle, WA, USA

## Funding Agency:

* * *

### I. Introduction

Optical flow estimation is a classic vision task that captures precise motion field in real time. Through generating an output optical flow field with the same size of input, per-pixel velocity could be obtained and then be utilized in downstream tasks, such as ego-motion estimation, segmentation, simultaneous localization and mapping (SLAM) [1], [2], [3]. On one hand, with the emerging of edge intelligent scenarios that have limited power budget, for example, autonomous driving or flying, embodied intelligence, etc., providing an efficient optical flow estimation becomes more than essential. On the other hand, these edge scenarios require a high-speed and accurate motion capturing, which can hardly be achieved by traditional frame-based cameras due to their fixed/low temporal resolution [4]. Therefore, event-based cameras with low power consumption and high temporal resolution, such as Dynamic Vision Sensors (DVS), have drawn widespread attention in optical flow estimation [5], [6].
