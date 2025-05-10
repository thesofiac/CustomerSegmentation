import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def treat_columns(df):
  df = df.dropna()

  # If buys on campaign
  df['Buys_on_campaign'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
  df = df.drop(['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'], axis=1)

  # Buys with discount
  df['Num_purchases'] = df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumWebPurchases']
  
  df = df[df['Num_purchases'] != 0]
  df = df[df['NumDealsPurchases'] < df['Num_purchases']]
  
  df['Purchases_with_descount'] = df['NumDealsPurchases']/df['Num_purchases']*100
  df['Purchases_with_descount'] = df['Purchases_with_descount'].clip(upper=79)
  
  bins = [0, 20, 40, 60, 80]
  labels = ['0-20', '20-40', '40-60', '60-80']
  df['Purchases_with_descount'] = pd.cut(df['Purchases_with_descount'], bins=bins, labels=labels, right=False)
  df['Purchases_with_descount'] = df['Purchases_with_descount'].replace({
      '0-20' : 0,
      '20-40' : 1,
      '40-60' : 2,
      '60-80' : 3
      }).astype(int)
      
  # Buys_where
  df['Catalog'] = df['NumCatalogPurchases']/df['Num_purchases']
  df['Store'] = df['NumStorePurchases']/df['Num_purchases']
  df['Web'] = df['NumWebPurchases']/df['Num_purchases']
  
  df['Preference'] = df[['Catalog', 'Store', 'Web']].idxmax(axis=1)
  df['Preference'] = df['Preference'].replace({
      'Catalog' : 0,
      'Store' : 1,
      'Web' : 2
      }).astype(int)
      
  df = df.drop(['Num_purchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebPurchases', 'NumDealsPurchases', 'Catalog', 'Store', 'Web'], axis=1)

  # How long is client
  df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
  today = df['Dt_Customer'].max()
  
  df['Is_client_since'] = (today - df['Dt_Customer']).dt.days
  bins = [0, 275, 550, 825, 1100]
  labels = ['0-275', '275-550', '550-825', '825-1100']
  df['Is_client_since'] = pd.cut(df['Is_client_since'], bins=bins, labels=labels, right=False)
  df['Is_client_since'] = df['Is_client_since'].replace({
      '0-275' : 0,
      '275-550' : 1,
      '550-825' : 2,
      '825-1100' : 3
      }).astype(int)
      
  df = df.drop('Dt_Customer', axis=1)

  # Is the client recently buying or not
  bins = list(range(0, 101, 25))
  labels = [f"{i}-{i+25}" for i in bins[:-1]]
  
  df['Recency'] = pd.cut(df['Recency'], bins=bins, labels=labels, right=False)
  df['Is_buying'] = df['Recency'].replace({
      '0-25' : 0,
      '25-50' : 1,
      '50-75' : 2,
      '75-100' : 3
      }).astype(int)
      
  df = df.drop('Recency', axis=1)

  # Family size
  df['Num_adults'] = df['Marital_Status'].replace({
      'Married' : 2,
      'Together' : 2,
      'Single' : 1,
      'Divorced' : 1,
      'Widow' : 1,
      'Alone' : 1,
      'Absurd' : 1,
      'YOLO' : 1
      }).astype(int)
      
  df['Num_kids'] = df['Kidhome'] + df['Teenhome']
  df['Family_size'] = df['Num_adults'] + df['Num_kids']
  
  # Have kids or not
  df['Is_kids'] = np.where(df['Num_kids'] == 0, 0, 1)
  
  # Amount spent per person
  df['Amount_spent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
  
  df['Amount_spent_per_person'] = df['Amount_spent']/df['Family_size']
  df['Amount_spent_per_person'] = df['Amount_spent_per_person'].clip(upper=699.99)
  bins = [0, 175, 350, 525, 700]
  labels = ['0-175', '175-350', '350-525', '525-700']
  df['Amount_spent_per_person'] = pd.cut(df['Amount_spent_per_person'], bins=bins, labels=labels, right=False)
  df['Amount_spent_per_person'] = df['Amount_spent_per_person'].replace({
      '0-175' : 0,
      '175-350' : 1,
      '350-525' : 2,
      '525-700' : 3
      }).astype(int)
      
  # Amount spent vs Income
  df = df[df['Income'] <= 200000]
  df = df.copy()
  df['Spent_vs_income'] = df['Amount_spent']*100/df['Income']
  df['Spent_vs_income'] = df['Spent_vs_income'].clip(upper=3.99)
  bins = [0, 1, 2, 3, 4]
  labels = ['0-1', '1-2', '2-3', '3-4']
  df['Spent_vs_income'] = pd.cut(df['Spent_vs_income'], bins=bins, labels=labels, right=False)
  df['Spent_vs_income'] = df['Spent_vs_income'].replace({
      '0-1' : 0,
      '1-2' : 1,
      '2-3' : 2,
      '3-4' : 3 
      }).astype(int)
      
  # Drinks or not
  df['Drinks'] = np.where(df['MntWines'] == 0, 0, 1)
  
  to_drop = ['Marital_Status', 'Kidhome', 'Teenhome', 'Num_kids', 'Num_adults',  'Amount_spent', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Income']
  df = df.drop(to_drop, axis=1)

  # Age from clients
  df = df[df['Year_Birth'] >= 1900]
  df = df.copy()
  
  bins = list(range(1900, 2001, 25))
  labels = [f"{i}-{i+25}" for i in bins[:-1]]
  
  df['Year_Birth'] = pd.cut(df['Year_Birth'], bins=bins, labels=labels, right=False)
  
  df['Year_Birth'] = df['Year_Birth'].replace({
      '1900-1925' : 0,
      '1925-1950' : 1,
      '1950-1975' : 2,
      '1975-2000' : 3
      }).astype(int)

  df['Education'] = df['Education' ].replace({
      'Basic' : 0,
      '2n Cycle' : 1,
      'Graduation' : 2,
      'Master' : 3,
      'PhD' : 4
      }).astype(int)

  ids = df['ID']
  df = df.drop(['ID', 'NumWebVisitsMonth', 'Z_CostContact', 'Z_Revenue'], axis=1)
  
  return df, ids


def scale_columns(df1):
  df2 = df1.copy()
  df2['Year_Birth'] = df2['Year_Birth']
  df2['Education'] = df2['Education']
  df2['Complain'] = df2['Complain']
  df2['Buys_on_campaign'] = df2['Buys_on_campaign']
  df2['Purchases_with_descount'] = df2['Purchases_with_descount']
  df2['Preference'] = df2['Preference']
  df2['Is_client_since'] = df2['Is_client_since']
  df2['Is_buying'] = df2['Is_buying'] * 5
  df2['Family_size'] = df2['Family_size']
  df2['Is_kids'] = df2['Is_kids'] * 5
  df2['Amount_spent_per_person'] = df2['Amount_spent_per_person']
  df2['Spent_vs_income'] = df2['Spent_vs_income']
  df2['Drinks'] = df2['Drinks'] * 5

  return df2

# Carregar modelo e dados
pipeline = joblib.load('kmeans_pipeline.pkl')

# Clusterizar dados originais
original_df = pd.read_csv('marketing_campaign.csv', sep='\t')
original_treated, ids = treat_columns(original_df)
original_scaled = scale_columns(original_treated)

original_labels = pipeline.predict(original_scaled)
original_scaled['cluster'] = original_labels
original_scaled['ID'] = ids

descricao_clusters = {
            1: "Grupo 1: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias médias, com crianças, constumam gastar pouco por integrante familiar e reservam pouco da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Respondem bem a campanhas promocionais, preferem fazer compras quando há algum tipo de desconto e têm comprado recentemente.  Têm o costume de comprar bebidas alcoolicas. \n Estratégias de venda: é um grupo de clientes já fiel, que tem feito compras recentes. A oferta de cupons de desconto e veiculação de campanhas promocionais para esse grupo de clientes deve gerar bons resultados. Itens infantis e bebidas alcoolicas com desconto podem ser ofertadas.",
            2: "Grupo 2: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias médias, com crianças, constumam gastar pouco por integrante familiar e reservam pouco da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Não respondem bem a campanhas promocionais, preferem fazer compras quando há algum tipo de desconto e faz bastante tempo que não têm comprado na loja.  Têm o costume de comprar bebidas alcoolicas. \n Estratégias de venda: por ser um grupo que está há muito sem comprar na loja, mas prioriza a compras com descontos, a oferta de cupons de desconto pode ser atrativa. Campanhas promocionais geram poucos resultados nesse grupo. Programas de fidelidade os incentivarão a voltar a comprar. Itens infantis e bebidas alcoolicas com desconto podem ser ofertadas.",
            3: "Grupo 3: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias pequenas, sem crianças, constumam gastar um valor de médio a alto por integrante familiar e reservam valor médio a pequeno da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Respondem bem a campanhas promocionais, costumam comprar sem desconto e têm comprado recentemente. Têm o costume de comprar bebidas alcoolicas. \n Estratégias de venda: é um grupo de alta renda, que, apesar de gastar bastante por integrante familiar, as compras não comprometemsignificativamente a renda familiar. Há margem para expansão das vendas com a realização de campanhas promocionais. A oferta de cupons de desconto não é necessária, já que este grupo costuma comprar mesmo sem desconto. Consomem bebidas alcoolicas, mas não itens infantis.",
            4: "Grupo 4: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias médias, com crianças, constumam gastar pouco por integrante familiar e reservam pouco da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Não respondem bem a campanhas promocionais, preferem fazer compras quando há algum tipo de desconto e têm comprado recentemente. Têm o costume de comprar bebidas alcoolicas. \n Estratégias de vendas: Campanhas promocionais geram poucos resultados nesse grupo, mas a oferta de cupons de desconto pode ser atrativa. Itens infantis e bebidas alcoolicas com desconto podem ser ofertadas.",
            5: "Grupo 5: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias médias, com crianças, constumam gastar pouco por integrante familiar e reservam pouco da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Não respondem bem a campanhas promocionais, preferem fazer compras quando há algum tipo de desconto e faz bastante tempo que não têm comprado na loja. Têm o costume de comprar bebidas alcoolicas. \n Estratégias de vendas: Campanhas promocionais geram poucos resultados nesse grupo, mas a oferta de cupons de desconto pode ser atrativa. Itens infantis e bebidas alcoolicas com desconto podem ser ofertadas. Estão sem comprar na loja há um tempo, então programas de fidelidade os incentivarão a voltar a comprar. Itens infantis e bebidas alcoolicas com desconto podem ser ofertadas.",
            6: "Grupo 6: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias pequenas, sem crianças, constumam gastar muito por integrante familiar e reservam um valor médio da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Respondem bem a campanhas promocionais, costumam comprar sem desconto e faz bastante tempo que não têm comprado na loja. Têm o costume de comprar bebidas alcoolicas. \n Estratégias de vendas: é um grupo de renda alta, que já gasta bastante por integrante familiar e reserva valor médio da renda familiar para compras na loja. Não há tanta margem para mais vendas, mas respondem bem a campanhas promocionais. A oferta de cupons de desconto não é necessária, já que este grupo costuma comprar mesmo sem desconto. Estão sem comprar na loja há um tempo, então programas de fidelidade os incentivarão a voltar a comprar. Itens infantis e bebidas alcoolicas com desconto podem ser ofertadas. Consomem bebidas alcoolicas, mas não itens infantis.",
            7: "Grupo 7: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias pequenas, sem crianças, constumam gastar muito por integrante familiar e reservam um valor médio da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Respondem muito bem a campanhas promocionais, costumam comprar sem desconto e têm comprado recentemente. Têm o costume de comprar bebidas alcoolicas. \n Estratégias de vendas: é um grupo de renda alta, que já gasta bastante por integrante familiar e reserva valor médio da renda familiar para compras na loja. Não há tanta margem para mais vendas, mas respondem muito bem a campanhas promocionais, comprando em quase todas elas. A oferta de cupons de desconto não é necessária, já que este grupo costuma comprar mesmo sem desconto. Consomem bebidas alcoolicas, mas não itens infantis.",
            8: "Grupo 8: Grupo de pessoas com idades e níveis de escolaridade variados. São de famílias pequenas, sem crianças, constumam gastar muito por integrante familiar e reservam um valor médio da renda familiar para compras em nossa loja. Compram tanto no site, quanto na loja física e pelo catálogo. Respondem bem a campanhas promocionais, costumam comprar sem desconto e faz bastante tempo que não têm comprado na loja. Têm o costume de comprar bebidas alcoolicas. \n Estratégias de vendas: é um grupo de renda alta, que já gasta bastante por integrante familiar e reserva valor médio da renda familiar para compras na loja. Não há tanta margem para mais vendas, mas respondem muito bem a campanhas promocionais, comprando em quase todas elas. A oferta de cupons de desconto não é necessária, já que este grupo costuma comprar mesmo sem desconto. Estão sem comprar na loja há um tempo, então programas de fidelidade os incentivarão a voltar a comprar. Consomem bebidas alcoolicas, mas não itens infantis."}

st.set_page_config(layout="wide")
st.title("Classificador Binário")

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Busque os dados de um grupo",
    "Busque grupos por característica",
    "Preveja a qual grupo um cliente pertence",
    "Busque um cliente por ID",
    "Entenda a escolha do modelo"
])

dic_scholarity = {
        'Ensino Fundamental' : 'Basic',
        'Ensino Médio' : '2n Cycle',
        'Graduação' : 'Graduation',
        'Mestrado' : 'Master',
        'Doutorado' : 'PhD'
        }

dic_marital_status = {
        'Solteiro' : 'Single',
        'Casado' : 'Married',
        'Divorciado' : 'Divorced',
        'Viúvo' : 'Widow'
        }

dic_binary = {
        'Sim' : '1' ,
        'Não' : '0'
        }

scholarity_options = list(dic_scholarity.keys())
marital_options = list(dic_marital_status.keys())
binary_options = list(dic_binary.keys())


if menu == "Entenda os dados":
    st.subheader("Entenda os dados")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: justify'><h5>Neste projeto, temos por objetivo determinar se um candidato a uma vaga de cientista de dados será propenso a trocar de emprego após realizar o treinamento oferecido pela empresa ou não. Como é do interesse das empresas evitar a contratação de candidatos que as deixarão após o treinamento, podemos tratar esse problema como um caso de Classificação Binária. Ou seja, um caso em que o <i>target</i> é classificatório e binário do tipo False (não deixará a empresa) e True (deixará a empresa). Para que seja determinada a probabilidade do candidato estar interessado em mudar de emprego, foram analisados dados de gênero, formação e experiência profissional, assim como dados das vagas e empresas que estes profissionais ocupavam no momento da coleta de dados.</h5></div>", unsafe_allow_html=True)

elif menu == "Busque os dados de um grupo":
    # Título
    st.subheader("Busque os dados de um grupo")

    # Entrada do usuário
    cluster_number = st.number_input("Digite um número de 1 a 8:", min_value=1, max_value=8, step=1)
    
    # Mostrar resultado
    if cluster_number in descricao_clusters:
      st.success(descricao_clusters[cluster_number])
    else:
      st.warning("Escolha um número entre 1 e 8.")
      
elif menu == "Preveja a qual grupo um cliente pertence":
    st.subheader("Preveja a qual grupo um cliente pertence")

    f1 = '0'  # ID
    f2 = st.number_input("Ano de Nascimento", min_value=0, step=1)  # Birth
    f3 = st.selectbox("Escolaridade", scholarity_options)  # Escolaridade
    f4 = st.selectbox("Estado Civil", marital_options)  # Estado civil
    f5 = st.number_input("Renda Familiar", min_value=0, value=0.0)  # Renda
    f6 = st.number_input("Número de Crianças na Família", min_value=0, step=1)  # Nº de crianças
    f7 = st.number_input("Número de Adolescentes na Família", min_value=0, step=1)  # Nº de adolescentes

    f8 = st.text_input("Digite a data da primeira compra do cliente (DD-MM-YYYY)")  # Primeira compra
    f9 = st.number_input("Dias desde sua última compra", min_value=0, step=1)  # Dias desde sua última compra

    f10 = st.number_input("Gastos com bebida alcoolica", min_value=0, value=0.0)  # Bebidas
    f11 = st.number_input("Gastos com frutas", min_value=0, value=0.0)  # Frutas
    f12 = st.number_input("Gastos com carnes", min_value=0, value=0.0)  # Carnes
    f13 = st.number_input("Gastos com peixes", min_value=0, value=0.0)  # Peixes
    f14 = st.number_input("Gastos com doces", min_value=0, value=0.0)  # Doces
    f15 = st.number_input("Gastos com joias", min_value=0, value=0.0)  # Joias

    f16 = st.number_input("Número de compras com descontos", min_value=0, step=1)  # Descontos
    f17 = st.number_input("Número de compras realizadas no site", min_value=0, step=1)  # Web
    f18 = st.number_input("Número de compras realizadas pelo catálogo", min_value=0, step=1)  # Catálogo
    f19 = st.number_input("Número de compras realizadas na loja física", min_value=0, step=1)  # Loja física
    f20 = st.number_input("Número visitas realizadas ao site", min_value=0, step=1)  # Visitas

    f21 = st.selectbox("Realizou compras durante a primeira campanha?", binary_options)  # Campanha 1
    f22 = st.selectbox("Realizou compras durante a segunda campanha?", binary_options)  # Campanha 2
    f23 = st.selectbox("Realizou compras durante a terceira campanha?", binary_options)  # Campanha 3
    f24 = st.selectbox("Realizou compras durante a quarta campanha?", binary_options)  # Campanha 4
    f25 = st.selectbox("Realizou compras durante a quinta campanha?", binary_options)  # Campanha 5
    f26 = st.selectbox("Realizou compras durante a última campanha?", binary_options) # Response

    f27 = st.selectbox("Realizou reclamações nos últimos dois anos?", binary_options) # Reclamações
    f28 = '0'  # Z_cost
    f29 = '0'  # Z_revenue

    if st.button("Prever"):
        input_df = pd.DataFrame([[int(f1), int(f2), dic_scholarity[f3], dic_marital[f4], float(f5), int(f6), int(f7), str(f8), int(f9), int(f10), int(f11), int(f12), int(f13), int(f14), int(f15), int(f16), int(f17), int(f18), int(f19), int(f20), int(f21), int(f22), int(f23), int(f24), int(f25), int(f27), int(f28), int(f29), int(f26)]], columns=['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response'])

        treated_input_df, ids = treat_columns(input_df)
        input_df_scaled = scale_columns(treated_input_df)

        input_label = pipeline_loaded.predict(input_df_scaled)
        label = int(input_label) + 1
        input_df['cluster'] = input_label
        input_df['ID'] = ids

        st.write(f"O grupo ao qual o cliente pertence é: grupo {label}")
        texto_explicativo = descricao_clusters.get(cluster, "Descrição não disponível para este grupo.")
        st.markdown(f"Características do grupo {label}: ")
        st.write(texto_explicativo)


elif menu == "Busque grupos por característica":
    # Título
    st.subheader("Busque grupos por característica")

    # Mapeamento de frases para clusters
    descricao_para_clusters = {
            "Clientes com crianças na família": [1, 2, 4, 5],
            "Clientes que respondem bem a campanhas promocionais": [1, 3, 6, 7, 8],
            "Clientes que preferem compras com desconto": [1, 2, 4, 5],
            "Clientes que estão sem comprar há muito tempo": [2, 5, 6, 8],
            "Clientes que bebem": [1, 2, 3, 4, 5, 6, 7, 8]
            }

    # Interface do Streamlit
    opcao = st.selectbox("Escolha a característica do grupo:", list(descricao_para_clusters.keys()))
    clusters_escolhidos = descricao_para_clusters[opcao]

    lista_clusteres = []
    for i in clusters_escolhidos:
      l = i-1
      lista_clusteres.append(l)
      
    if opcao:
        resultados = original_scaled[original_scaled['cluster'].isin(lista_clusteres)]
        lista = resultados['ID'].tolist()
      
        st.write(f"Grupos correspondentes: {clusters_escolhidos}")
        st.write("IDs dos clientes encontrados:")
        st.dataframe(pd.DataFrame(sorted(lista), columns=['ID']))


elif menu == "Busque um cliente por ID":
    # Título
    st.subheader("Busque um cliente por ID")

    # Entrada do usuário
    id_cliente = st.number_input("Digite o ID do cliente:", min_value=0, step=1)

    # Input do usuário
    id_cliente = st.number_input("Digite o ID do cliente:", min_value=0, step=1)

    if id_cliente in original_scaled['ID'].values:
        cluster = df_final[original_scaled['ID'] == id_cliente]['cluster'].iloc[0]
        st.success(f"O cliente {id_cliente} pertence ao cluster {cluster}")

        # Mostra a explicação
        texto_explicativo = descricao_clusters.get(cluster, "Descrição não disponível para este cluster.")
        st.markdown(f"Características do cluster {cluster}: ")
        st.write(texto_explicativo)

    else:
        st.warning("ID não encontrado no conjunto de dados.")


elif menu == "Entenda a escolha do modelo":
    st.subheader("Entenda a escolha do modelo")
    st.markdown("""<div style='text-align: justify;'><h5>Neste projeto, temos por objetivo determinar se um candidato a uma vaga de cientista de dados será propenso a trocar de emprego após realizar o treinamento oferecido pela empresa ou não. Como é do interesse das empresas evitar a contratação de candidatos que as deixarão após o treinamento, podemos tratar esse problema como um caso de <b>Classificação Binária</b>. Ou seja, um caso em que o <i>target</i> é binário: False (não deixará a empresa) ou True (deixará a empresa).<br><br>
        Para selecionar o melhor modelo e a metodologia a serem aplicados aos dados, inicialmente foram avaliadas a presença de valores nulos, outliers e erros de grafia. Os valores nulos foram tratados por meio da substituição pela média, mediana ou moda, a depender do tipo e da distribuição das variáveis.<br><br>
        Outliers foram considerados como valores numéricos acima de 1 amplitude interquartílica do terceiro quartil ou abaixo de 1 amplitude interquartílica do primeiro quartil.<br><br>
        As colunas numéricas nominais foram escaladas utilizando o <i>StandardScaler</i>.<br><br>
        Como foi identificado um forte desbalanceamento nas classes do <i>target</i> — com aproximadamente 75% de dados False e 25% True — quatro técnicas de balanceamento foram testadas: SMOTE, ADASYN, Random Over-Sampling e Random Under-Sampling.<br><br>
        Os modelos de classificação avaliados inicialmente foram: <i>LogisticRegression</i>, <i>RandomForestClassifier</i> e <i>XGBClassifier</i>, testados em uma gama de hiperparâmetros definida por meio do <i>GridSearchCV</i>.<br><br>
        Variáveis constantes foram removidas com base em sua variância, utilizando o <i>VarianceThreshold</i>, e as 10 variáveis mais relevantes foram selecionadas por meio do <i>SelectKBest</i>.<br><br>
        O <b>recall</b> foi adotado como métrica principal de avaliação dos modelos, pois, de acordo com os objetivos do problema, é mais importante identificar o maior número possível de funcionários propensos a pedir demissão (targets True) do que prever corretamente os casos negativos.<br><br>
        O modelo com melhor desempenho foi o <i>RandomForestClassifier</i> com balanceamento via SMOTE, atingindo recall de 74% no treino e 73% no teste.<br><br>
        Considerando o desbalanceamento original dos dados, esse desempenho foi considerado satisfatório, já que, mesmo com apenas 25% de casos positivos, o modelo conseguiu identificar corretamente 73% deles no conjunto de teste.<br><br>
        O código utilizado pode ser encontrado no <a href='https://github.com/thesofiac/Streamlit-JobChange-Analysis' target='_blank'>repositório do GitHub</a>.</h5></div>""", unsafe_allow_html=True)
