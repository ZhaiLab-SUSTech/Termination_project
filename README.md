# pipeline of analyzing transcript termination 

## 0.Preparation
### packages install

- Python version
Python 3.8.18 and above


- recommand using conda to install all packages

  - `conda create -n terminate_env python==3.8`
  - `conda activate terminate_env`
  - `conda install -c bioconda -c conda-forge matplotlib scipy numpy pandas pysam seaborn cycler` 
  - `pip install -r main_req.txt`



- This command will create a enviroment: `termination_env`

- FLEPSeq enviroment is required
  https://github.com/ZhaiLab-SUSTech/flep_seq2_polya_analysis

- Termination_landscape enviroment is required
  https://github.com/ZhaiLab-SUSTech/Termination_landscape

## 1 Nanopore bascalling and obtain the treanscript termination intermediates

### 1.1 Basecalling
```shell
guppy_basecaller -i {fast5_dir} -s {pass_fast5_dir} -c {model} --recursive --fast5_out --disable_pings --qscore_filtering --device {params_cuda}
```

- `{fast5_dir}: directory of raw fast5`
- `{pass_fast5_dir}: directory of output fast5`
- `{model}: type of sample (DNA or RNA) experimental apparatus ,reagents. ex. dna_r9.4.1_450bps_hac.cfg`
- `{params_cuda}: GPU and cuda version. ex. "cuda:all:100%"`

- `tips: Parameters may be different for different versions of guppy, so please pay attention to modify them.`



### 1.2 Transfer fastq to fasta
```shell
python fastqdir2fasta.py --indir {fastq_dir} --out {fasta_file}
```

- `{fastq_dir}: directory of fastq, generated by bascalling.`
- `{fasta_file}: output fasta file, recommand to store this in new a directory.`



### 1.3 Mapping process
```shell
minimap2 -t {threads} -ax splice --secondary=no -G 12000 {genome} {fasta_file} -o {mediate_sam}
samtools view -@ {threads} -F 2308 -hb {mediate_sam} {mediate_bam}
samtools sort -@ {threads} -O bam -o {sort_bam} {mediate_bam}
samtools index -@ {threads} {sort_bam}
```

- `{threads}: The number of threads`
- `{genome}: The genome of fasta format`

- `tips: Parameters may be different for different versions of samtools. The sorted bam will generate, and you can delete the sam file after checking that the output file is correct.`



### 1.4 Find 3'linker
```shell
python adapterFinder.py --inbam {input_bam} --inseq {fasta_file} --out {adapter} --threads {threads} --mode 1
```

- `{adapter}: File contain the adapter's relative location and other infromation on coressponding read.`



### 1.5 Identify and estimate the poly(A) tail
```shell
python PolyACaller.py --inadapter {adapter} --summary {sequencing_summary}  --fast5dir {pass_fast5_dir} --out {polyA_tail} --threads {threads}
```

- `{sequencing_summary}: Sequencing summary file is generated by basecalling step.`
- `{polyA_tail}: Output result of poly(A) tail result.`



### 1.6 Extract read information
```shell
python extract_read_info.py --inbed {bed} --inbam {sort_bam} --out {read_info}
```

- `{read_info}: Extraction of information based on read and relative position of exons and introns on the genome.`



### 1.7 Add tag to bam
```shell
python add_tag_to_bam.py -i {sort_bam} --read_info {read_info} --adapter_info {adapter} --polya_info {polyA_tail} -o {tag_sort_bam}
```

- `{tag_sort_bam}: bam file with the additional tags, inclues the reads with/without poly(A) tail, gene_id et al.`



### 1.8 Separate reads to elongation and polyadenylated group
```shell
python get_elongating_reads.py -i {tag_sort_bam} -l 15 -o {raw_elongating_bam}
samtools view -@ 2 -h -bq 1 {raw_elongating_bam} > {elongating_bam}

python get_polyadenylated_reads.py -i {tag_sort_bam} -l 15 -o {raw_polyadenylated_bam}
samtools view -@ 2 -h -bq 1 {raw_polyadenylated_bam} > {polyadenylated_bam}
```

- `{elongating_bam}: Elongating transcripts were identified by predicted poly(A) tail length <15 and used for further analysis`
- `{polyadenylated_bam}: Polyadenylated transcripts were identified by predicted poly(A) tail length ≥15 and used for further analysis`



### 1.9 Separate elongation reads to different intermediates transcripts

- bam_list.txt contains all elongation bam files

```text
sample ../demo_data/sample_elongation.bam
...
```

#### 1.9.1 generate a bam list
```sh
while read line
do
    sample_name=$(echo "$line" | awk '{print $1}')
    bam_file=$(echo "$line" | awk '{print $2}')
    python split_bam_to_intermediates.py -s $sample_name \
        -c 1 \
        -o /seperate_intermediates/ \
        -b $bam_file &
    wait
done < bam_list.txt
wait 
```
- This script will generate intermediates transcripts
    `readthrough_cl.bam`
    `five_cl.bam`
    `three_cl.bam`

#### 1.9.2 merge readthrough and 3' cleavage products

```shell
out_fold=/seperate_intermediates

while read line
do
    sample_name=$(echo "$line" | awk '{print $1}')
    cl3=${out_fold}/${sample_name}/three_cl.bam
    rt=${out_fold}/${sample_name}/readthrough_cl.bam
    rt_cl3=${out_fold}/${sample_name}/readthrough_three_merge.cl.bam
    samtools merge -@ 2 -o ${rt_cl3} ${cl3} ${rt} &
done < bam_list.txt
wait
```

#### 1.9.3 generate index

```shell
while read line
do
    sample_name=$(echo "$line" | awk '{print $1}')
    samtools index ${out_fold}/${sample_name}/five_cl.bam &
    samtools index ${out_fold}/${sample_name}/three_cl.bam &
    samtools index ${out_fold}/${sample_name}/readthrough_cl.bam &
    samtools index ${out_fold}/${sample_name}/readthrough_three_merge.cl.bam &
done < bam_list.txt
wait
```



## 2 transcrition termination analysis by jupyterlab
- `run Step2.calculate_TW.ipynb in the jupyterlab enviroment`
