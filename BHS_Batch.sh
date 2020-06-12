#!/bin/bash  

#SBATCH --job-name=Test_BHS
#SBATCH --account=********** #adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=4:00:00   #adjust this to match the walltime of your job
#SBATCH --nodes=1    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #adjust this if you are using PCT
#SBATCH --mem=32G     #adjust this according to your the memory requirement per node you need
#SBATCH --mail-type=ALL

python PlotXval2.py
