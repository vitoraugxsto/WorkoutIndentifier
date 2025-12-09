import pandas as pd

# Dicionário com nomes de arquivos e seus identificadores
arquivos = {
    "rosca1_rotulado.csv": "rosca1",
    "rosca2_rotulado.csv": "rosca2",
    "rosca3_rotulado.csv": "rosca3",
    "rosca4inc_rotulado.csv": "rosca4inc",
    "rosca5cor_rotulado.csv": "rosca5cor"
}

dfs = []

print("🔍 Verificando arquivos CSV...")
for nome_arquivo, identificador in arquivos.items():
    print(f"\n📄 Lendo {nome_arquivo}...")
    df = pd.read_csv(nome_arquivo)
    df["video"] = identificador

    if "movimento_incorreto" not in df.columns:
        print(f"⚠️  Coluna 'movimento_incorreto' **NÃO encontrada** em {nome_arquivo}. Será criada com valor padrão False.")
        df["movimento_incorreto"] = False
    else:
        print(f"✅ Coluna 'movimento_incorreto' encontrada em {nome_arquivo}.")

    dfs.append(df)

# Concatenar todos os DataFrames
df_total = pd.concat(dfs, ignore_index=True)

# Criar coluna 'label'
df_total["label"] = df_total["movimento_incorreto"].astype(int)

# Salvar
df_total.to_csv("dataset_rosca_completo.csv", index=False)

print("\n✅ Dataset unificado criado como 'dataset_rosca_completo.csv'")
print(f"🔢 Total de registros: {len(df_total)}")
print(df_total.head())
