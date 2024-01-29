
# Estudo: Busca Semântica em Dados de Texto

Este é um projeto Python que utiliza a biblioteca SentenceTransformer para realizar busca semântica em dados de texto. O projeto lê arquivos de um diretório, divide o texto em pedaços e indexa os pedaços em um banco de dados PostgreSQL usando a extensão pgvector para busca de similaridade de vetores.

## Começando

Estas instruções irão fornecer uma cópia do projeto em funcionamento na sua máquina local para fins de desenvolvimento e teste.

### Pré-requisitos

- Python 3.8 ou superior
- PostgreSQL
- Extensão pgvector para PostgreSQL

### Instalando

1. Clone o repositório
2. Instale os pacotes Python necessários usando pip:

```bash
pip install -r requirements.txt
```

3. Configure um banco de dados PostgreSQL e instale a extensão pgvector.

### Uso

As principais funções do projeto são `indexar` e `buscar`. A função `indexar` lê arquivos de um diretório especificado, divide o texto em pedaços e indexa os pedaços no banco de dados PostgreSQL. A função `buscar` realiza uma busca semântica nos dados indexados.

```python
if __name__ == '__main__':
    run_indexar()
    buscar('Sua consulta aqui')
```

## Construído com

- [Python](https://www.python.org/)
- [PostgreSQL](https://www.postgresql.org/)
- [pgvector](https://github.com/ankane/pgvector)
- [SentenceTransformer](https://www.sbert.net/)
