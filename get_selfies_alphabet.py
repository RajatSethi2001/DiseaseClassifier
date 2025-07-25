import pandas as pd
import selfies as sf

def main():
    compound_file = "Data/compoundinfo_beta.txt"
    compound_df = pd.read_csv(compound_file, sep="\t")
    smiles_list = compound_df["canonical_smiles"].to_list()
    selfies_list = []
    for smiles in smiles_list:
        try:
            selfies = sf.encoder(smiles)
            selfies_list.append(selfies)
        except:
            continue
    
    selfies_alphabet = list(sf.get_alphabet_from_selfies(selfies_list))
    selfies_alphabet.append(".")
    selfies_alphabet.append("[SKIP]")
    selfies_alphabet.sort()

    with open("Data/selfies_alphabet.txt", "w") as f:
        for token in selfies_alphabet:
            f.write(f"{token}\n")
    
    print(selfies_alphabet)
    
if __name__=="__main__":
    main()