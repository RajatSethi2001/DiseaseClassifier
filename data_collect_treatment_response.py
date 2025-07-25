import json
import subprocess

def run_cmd(cmd):
    print(f"Running Command: {cmd}")
    process = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end="")
    
    process.wait()

#Pre-treatment SRA, Post-treatment SRA, Drug SMILES
sra_data = [["SRR33743718", "SRR33743715", "Cc1c2oc3c(C)ccc(C(O)=N[C@@H]4C(O)=N[C@H](C(C)C)C(=O)N5CCC[C@H]5C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]4C)c3nc-2c(C(O)=N[C@@H]2C(O)=N[C@H](C(C)C)C(=O)N3CCC[C@H]3C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]2C)c(N)c1=O"],
            ["SRR33743717", "SRR33743714", "Cc1c2oc3c(C)ccc(C(O)=N[C@@H]4C(O)=N[C@H](C(C)C)C(=O)N5CCC[C@H]5C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]4C)c3nc-2c(C(O)=N[C@@H]2C(O)=N[C@H](C(C)C)C(=O)N3CCC[C@H]3C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]2C)c(N)c1=O"],
            ["SRR33743716", "SRR33743713", "Cc1c2oc3c(C)ccc(C(O)=N[C@@H]4C(O)=N[C@H](C(C)C)C(=O)N5CCC[C@H]5C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]4C)c3nc-2c(C(O)=N[C@@H]2C(O)=N[C@H](C(C)C)C(=O)N3CCC[C@H]3C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]2C)c(N)c1=O"],
            ["SRR33629977", "SRR33629974", "O=C1/N=C(\N=C/N1[C@@H]2O[C@@H]([C@@H](O)[C@H]2O)CO)N"],
            ["SRR33629976", "SRR33629973", "O=C1/N=C(\N=C/N1[C@@H]2O[C@@H]([C@@H](O)[C@H]2O)CO)N"],
            ["SRR33629975", "SRR33629972", "O=C1/N=C(\N=C/N1[C@@H]2O[C@@H]([C@@H](O)[C@H]2O)CO)N"],
            ["SRR33629971", "SRR33629968", "O=C1/N=C(\N=C/N1[C@@H]2O[C@@H]([C@@H](O)[C@H]2O)CO)N"],
            ["SRR33629970", "SRR33629967", "O=C1/N=C(\N=C/N1[C@@H]2O[C@@H]([C@@H](O)[C@H]2O)CO)N"],
            ["SRR33629969", "SRR33629966", "O=C1/N=C(\N=C/N1[C@@H]2O[C@@H]([C@@H](O)[C@H]2O)CO)N"],
            ["SRR33611224", "SRR33611221", "CC(=O)OCC[N+](C)(C)C"],
            ["SRR33611223", "SRR33611220", "CC(=O)OCC[N+](C)(C)C"],
            ["SRR33611222", "SRR33611219", "CC(=O)OCC[N+](C)(C)C"]]
parent_dir = "Treatment_Response"
spots_per_file = 5000000

run_cmd(f"mkdir -p {parent_dir}")
for data in sra_data:
    run_cmd(f"mkdir -p tmp")
    pre_treatment = data[0]
    post_treatment = data[1]
    smiles = data[2]

    data_dir = f"{parent_dir}/{pre_treatment}_{post_treatment}"
    run_cmd(f"mkdir -p {data_dir}")

    run_cmd(f"fastq-dump -N 1 -X {spots_per_file} --split-3 -O tmp {pre_treatment}")
    run_cmd(f"fastp -i tmp/{pre_treatment}_1.fastq -o tmp/{pre_treatment}_1_trim.fq -I tmp/{pre_treatment}_2.fastq -O tmp/{pre_treatment}_2_trim.fq -j tmp/fastp.json -h tmp/fastp.html")
    run_cmd(f"salmon quant -i salmon_index -l A -1 tmp/{pre_treatment}_1_trim.fq -2 tmp/{pre_treatment}_2_trim.fq -o tmp --validateMappings")
    run_cmd(f"python tx2gene.py -p tmp/quant.sf -o {data_dir}/pre_treatment.csv")

    run_cmd(f"fastq-dump -N 1 -X {spots_per_file} --split-3 -O tmp {post_treatment}")
    run_cmd(f"fastp -i tmp/{post_treatment}_1.fastq -o tmp/{post_treatment}_1_trim.fq -I tmp/{post_treatment}_2.fastq -O tmp/{post_treatment}_2_trim.fq -j tmp/fastp.json -h tmp/fastp.html")
    run_cmd(f"salmon quant -i salmon_index -l A -1 tmp/{post_treatment}_1_trim.fq -2 tmp/{post_treatment}_2_trim.fq -o tmp --validateMappings")
    run_cmd(f"python tx2gene.py -p tmp/quant.sf -o {data_dir}/post_treatment.csv")

    metadata = {
        "SMILES": smiles
    }

    with open(f"{data_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

    run_cmd(f"rm -rf tmp")





