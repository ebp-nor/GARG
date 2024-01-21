# GARG

Welcome to GARG **Genealogies and Ancestral Recombination Graphics**, a EBP-Nor workshop.


Brief workshop description (see below for a day-by-day description)
```
This is a comprehensive, 3-day program designed to introduce participants to the fascinating world of how recombination events shape genomes. This hands-on workshop will focus on utilizing the tskit environment, a powerful tool for simulating and analyzing genomic data (whole-genome resequencing data), to explore and understand complex inheritance patterns within populations.
```

Learning outcomes and competence
```
By the end of this 3-day workshop, participants will have gained valuable skills and knowledge in working with Ancestral Recombination Graphics using the tskit environment. They will be equipped to analyze, simulate, and interpret ARGs, contributing to their understanding of genetic ancestry and recombination events within populations.
```

Prerequisites
```
Familiarity with basic genetic concepts (alleles, mutations, genetic variation, etc.).
Basic knowledge of Python programming (though not mandatory, it will be helpful).
Participants are required to bring their laptops.
```

Dates
```
17-21 August 2024
```

Location of the workshop. [Here's the location on Googlemaps](https://www.google.com/maps/place/Storgata+43,+1440+Dr%C3%B8bak,+Norway/@59.6587149,10.6298365,20.64z/data=!4m9!1m2!2m1!1stollboden!3m5!1s0x46414424a65fcd75:0x7a9de006e9bd1221!8m2!3d59.6587603!4d10.6301111!16s%2Fg%2F11c26l0m8t?entry=ttu)
```
Drøbak research Station (Tollboden), Storgata 43, 1440 Drøbak, Norway.
```

Travel tips
```
Participants should arrive by their own means as we do not cover travel. If you arrive via Oslo, there is a public bus (number 500) that leaves from nearby Oslo S (central station) and takes about 40 minutes - 1 hour to reach Drøbak. This bus runs every hour. You can go to http://www.ruter.no, and set: From 'Oslo S', To 'Drøbak'. Google maps will also work. To get a ticket you need to download the 'Ruter app' on Playstore or the AppleStore.
```

Fees
```
This workshop is completely free, as it is sponsored by NORBIS and EBP-Nor. You only need to cover for your traveling. You will be hosted in the research station.
We have a max of 25 participants. We will select applicants based on an application page that will be made available.

```

Logistics (read this!)
```
We will be in the Tollboden research station - we will eat, sleep, and take classes here. It can feel very intensive but this is a very nice place with a pier where you can fish or swim (bring swimming gear). You will share room with 3-5 people (you can, alternatively, pay your own hotel room). You'll be given a towel and bedsheets (you'll make your own bed).

We will make a list of participants responsible for breakfast, lunch, and dinner. You are expected to contribute by setting up the table for breakfast and lunch and cooking dinner. We will have the support of the caretaker in buying food in the local supermarket.
```

Invited lecturers and organizers
```
Dr. Yan Wong (Oxford U, UK)
Dr. Mark Ravinet (UiO, Norway)
Dr. Per Unneberg (SciLab Uppsala, Sweden)
Gabriel David (Uppsala, Sweden)

Dr. José Cerca (UiO, Norway)
Dr. Ole K. Tørresen (UiO, Norway)
```

Workshop program
Day 0 (August 17th)

```
Arrival and introduction
Introductions of the participants and their projects.
Setting up the cluster.
**bring one slide ready**

```
Day 1 (18th of August) - Morning
```
Talk:
Introduction to different terms - Pedigrees, Genealogy, Historical Background, Different ARG Definitions
Introduction to Ancestral Recombination Graphs (ARGs) and their significance in studying genealogical history within populations.
Show how genetic data can be encoded on an ARG, neutrality, and the equivalence between site-based and branch-based population genetic statistics.
Discuss the various definitions of ARGs and approaches to constructing and simplifying them, setting the stage for a deeper understanding in subsequent parts of the workshop.

Practical:
Intro to tskit (edge-labelled gARGs)
Introduce the tskit environment, a powerful tool for handling and analyzing large-scale genetic data.
Focus on edge-labeled gARGs, a graphical representation within tskit that facilitates efficient analysis of ARGs.
Provide practical demonstrations to familiarize participants with using tskit for basic ARG analysis.
```

Day 1 (18th of August) - Afternoon
```
Talk & practical:
Simplification (as this is fundamental to forward simulation), collapsing of recombination nodes.
Build a very simple Wright-Fisher simulator. Explain the importance of simplifying ARGs for forward simulation and more efficient analysis.
Demonstrate techniques for collapsing recombination nodes to simplify ARGs without losing important genetic information.
Engage participants in practical exercises to simplify genealogies using tskit.

Talk:
Basic coalescent theory, coalescent with recombination, SPRs, etc.
Introduce participants to the fundamental principles of coalescent theory, a key concept in understanding genetic ancestry and population genetics.
Explain how the coalescent process is influenced by recombination events, resulting in Subtree Prune and Regraft (SPR) operations in local trees. Introduce the SMC and SMC’ (sequential Markov coalescent) approximations.
Discuss the implications of recombination on the genealogy of populations, preparing attendees for ARG analysis.

Analysis:
Single Site - Branch vs. Site Stats
Focus on analyzing single-site data within ARGs, comparing branch statistics with site statistics.
Discuss the insights that can be gained from examining genetic variation at specific loci within ARGs.
Illustrate the significance of single-site analysis in understanding recombination patterns.
Explore how haplotype-based approaches can contribute to ARG analysis.
```

Day 2 (19th of August) - Morning
```
Talk:
Simulating & Visualizing ARGs
Introduce techniques for simulating genealogies in forwards and backwards time, and the principle of recapitation.
Discuss how to simulate multiple chromosomes and the Wright-Fisher vs Hudson coalescent.
Explain how to record full ARGs during simulation to capture comprehensive genetic ancestry information.
Showcase methods to visualize simulated ARGs, helping participants grasp the patterns and complexity of genealogies.

Talk & Practical:
msprime (incl WF and Hudson models, multiple chromosomes, record_full_arg parameter, and likelihood calcs)
Dive into the msprime library, a powerful tool for simulating data.
Provide hands-on experience with running simulations, generating ARGs for multiple chromosomes, and recording full ARG information for analysis.
Introduce likelihood calculations, enabling participants to assess the fit between observed data and simulated ARGs.
```

Day 2 (19th of August) - Afternoon
```
Talk:
Dominance of Recombination Nodes if Unsimplified
Discuss the significance of recombination nodes and how their dominance affects ARG analysis.
Explain how unsimplified ARGs can impact forward simulations and the interpretation of genetic ancestry patterns.

Practicals:
Stdpopsim
Introduce Stdpopsim, a tool for generating standardized demographic models and simulating ARGs under different scenarios.
Discuss the advantages of using standardized demographic models in population genetics research.

SLiM
Explore the use of SLiM (Selection Linked to Individual-based Models) in simulating complex evolutionary scenarios, including selection and demographic events.
Highlight the versatility of SLiM in capturing various aspects of population genetics.
```

Day 3 (20th of August) - Morning
```
Talk
Inferring ARGs
Present different approaches for inferring ARGs from genetic data, emphasizing their strengths and limitations.
Discuss popular tools such as ARGweaver (MCMC), Relate (tree construction), ARGneedle (threading), and tsinfer (HMM matching).

Practical
Focus on tsinfer, incl. mismatch & SGkit
Focus on tsinfer as a powerful tool for inferring ARGs based on a Hidden Markov Model (HMM) matching approach.
Demonstrate the use of tsinfer for handling mismatched datasets and its integration with SGkit for enhanced analysis.
```

Day 3 (20th of August) - Afternoon
```
Talk
Tsdate
Introduce Tsdate, a method for dating ancestral recombination events within ARGs.
Explain how Tsdate can provide valuable insights into the timing of recombination events in evolutionary history.

Practical
Playtime with Real Datasets
Provide participants with real genetic datasets to work with, applying the techniques learned throughout the workshop.
Encourage exploration and analysis of ARGs from diverse populations and species to gain practical experience.
```

Day 4 (21 of August) - Breakfast and leaving.
