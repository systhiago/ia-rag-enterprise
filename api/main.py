import os
import re
import json
import time
import hashlib
import unicodedata
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import redis

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# Aplicação
# ============================================================

app = FastAPI(
    title="API LangChain com documentação prioritária, RAG, MCP e cache"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Configurações
# ============================================================

# URLs construídas desta forma para evitar que editores transformem
# automaticamente as URLs em tags HTML.
DEFAULT_OLLAMA_URL = "http" + "://" + "ollama:11434"
DEFAULT_MCP_URL = "http" + "://" + "mcp-server:3000"

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    DEFAULT_OLLAMA_URL
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://redis:6379"
)

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    DEFAULT_MCP_URL
).rstrip("/")

MCP_METRICS_ENDPOINT = os.getenv(
    "MCP_METRICS_ENDPOINT",
    "/metrics"
)

MCP_TIMEOUT_SECONDS = float(
    os.getenv("MCP_TIMEOUT_SECONDS", "5")
)

RAG_CACHE_TTL_SECONDS = int(
    os.getenv("RAG_CACHE_TTL_SECONDS", "86400")
)

POLICY_CACHE_TTL_SECONDS = int(
    os.getenv("POLICY_CACHE_TTL_SECONDS", "86400")
)

SEMANTIC_CACHE_THRESHOLD = float(
    os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.78")
)

DOCS_PATH = os.getenv(
    "DOCS_PATH",
    "/app/docs"
)


# ============================================================
# Clientes
# ============================================================

llm = OllamaLLM(
    base_url=OLLAMA_URL,
    model="llama3.1"
)

embeddings = OllamaEmbeddings(
    base_url=OLLAMA_URL,
    model="nomic-embed-text"
)

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)


# ============================================================
# Estado global
# ============================================================

global_vectorstore = None
documentacao_obrigatoria = ""
documentacao_hash = ""
documentos_carregados = []


# ============================================================
# Modelos
# ============================================================

class QueryRequest(BaseModel):
    question: str
    manual_prompt: str | None = None


# ============================================================
# Funções gerais
# ============================================================

def normalizar_texto(texto: str) -> str:
    texto = texto.lower().strip()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


def cosine_similarity(a, b) -> float:
    vetor_a = np.asarray(a, dtype=float)
    vetor_b = np.asarray(b, dtype=float)

    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return float(
        np.dot(vetor_a, vetor_b)
        / (norma_a * norma_b)
    )


def registrar_metrica(
    source: str,
    metric_type: str,
    question: str,
    context_used: str,
    header_size_bytes: int,
    execution_time_seconds: float,
    similarity_score: float | None = None,
    extra: dict | None = None
):
    metrica = {
        "source": source,
        "type": metric_type,
        "question": question,
        "context_used": context_used,
        "header_size_bytes": header_size_bytes,
        "execution_time_seconds": execution_time_seconds,
        "timestamp": time.time()
    }

    if similarity_score is not None:
        metrica["similarity_score"] = round(
            float(similarity_score),
            4
        )

    if extra:
        metrica.update(extra)

    redis_client.rpush(
        "lab_metrics",
        json.dumps(
            metrica,
            ensure_ascii=False
        )
    )


# ============================================================
# Carregamento prioritário da documentação
# ============================================================

def carregar_documentacao():
    global documentacao_obrigatoria
    global documentacao_hash
    global documentos_carregados

    if not os.path.exists(DOCS_PATH):
        print(
            f"ERRO: Diretório de documentação não encontrado: "
            f"{DOCS_PATH}"
        )

        documentacao_obrigatoria = ""
        documentacao_hash = ""
        documentos_carregados = []

        return False

    blocos_documentacao = []
    documentos = []
    arquivos_encontrados = []

    try:
        for root, _, files in os.walk(DOCS_PATH):
            for nome_arquivo in sorted(files):
                if not nome_arquivo.lower().endswith(".md"):
                    continue

                caminho = os.path.join(
                    root,
                    nome_arquivo
                )

                with open(
                    caminho,
                    "r",
                    encoding="utf-8"
                ) as arquivo:
                    conteudo = arquivo.read().strip()

                if not conteudo:
                    continue

                caminho_relativo = os.path.relpath(
                    caminho,
                    DOCS_PATH
                )

                bloco = (
                    f"\n\n===== DOCUMENTO: "
                    f"{caminho_relativo} =====\n"
                    f"{conteudo}\n"
                    f"===== FIM DO DOCUMENTO: "
                    f"{caminho_relativo} ====="
                )

                blocos_documentacao.append(bloco)

                documentos.append(
                    Document(
                        page_content=conteudo,
                        metadata={
                            "source": caminho,
                            "file": caminho_relativo
                        }
                    )
                )

                arquivos_encontrados.append(
                    caminho_relativo
                )

        documentacao_obrigatoria = "\n".join(
            blocos_documentacao
        ).strip()

        documentos_carregados = documentos

        documentacao_hash = hashlib.sha256(
            documentacao_obrigatoria.encode("utf-8")
        ).hexdigest()

        print(
            "SUCESSO: Documentação obrigatória carregada."
        )

        print(
            f"DEBUG: Arquivos .md carregados: "
            f"{len(arquivos_encontrados)}"
        )

        for arquivo in arquivos_encontrados:
            print(f"DEBUG: Documento carregado: {arquivo}")

        print(
            f"DEBUG: Hash da documentação: "
            f"{documentacao_hash}"
        )

        return bool(documentos)

    except Exception as erro:
        print(
            f"ERRO ao carregar documentação: {erro}"
        )

        documentacao_obrigatoria = ""
        documentacao_hash = ""
        documentos_carregados = []

        return False


# ============================================================
# Avaliação prioritária da documentação
# ============================================================

def criar_chave_validacao_documental(
    pergunta: str
) -> str:
    conteudo = (
        f"{documentacao_hash}:"
        f"{normalizar_texto(pergunta)}"
    )

    identificador = hashlib.sha256(
        conteudo.encode("utf-8")
    ).hexdigest()

    return f"policy_cache:{identificador}"


def extrair_json_modelo(resposta: str) -> dict:
    texto = resposta.strip()

    texto = re.sub(
        r"^```(?:json)?\s*",
        "",
        texto,
        flags=re.IGNORECASE
    )

    texto = re.sub(
        r"\s*```$",
        "",
        texto
    )

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        correspondencia = re.search(
            r"\{.*\}",
            texto,
            flags=re.DOTALL
        )

        if not correspondencia:
            raise ValueError(
                "O modelo não retornou um JSON válido."
            )

        return json.loads(
            correspondencia.group(0)
        )


def validar_documentacao(
    pergunta: str
) -> dict:
    """
    Essa função é obrigatoriamente executada antes de:
    - MCP
    - cache semântico de respostas
    - RAG
    - LLM de resposta

    A chave inclui o hash de todos os documentos. Portanto,
    qualquer alteração nos .md invalida automaticamente as
    decisões anteriores.
    """

    if not documentacao_obrigatoria:
        return {
            "permitido": False,
            "motivo": (
                "A documentação obrigatória não está disponível."
            ),
            "resposta": (
                "Não posso processar a solicitação porque as "
                "regras obrigatórias do diretório docs não "
                "foram carregadas."
            ),
            "source": "documentation_unavailable"
        }

    chave_cache = criar_chave_validacao_documental(
        pergunta
    )

    decisao_cache = redis_client.get(
        chave_cache
    )

    if decisao_cache:
        try:
            decisao = json.loads(
                decisao_cache
            )

            decisao["source"] = (
                "documentation_validation_cache"
            )

            return decisao

        except json.JSONDecodeError:
            redis_client.delete(
                chave_cache
            )

    prompt_validacao = ChatPromptTemplate.from_template(
        """
Você é o mecanismo obrigatório de validação documental
de uma aplicação local.

Leia integralmente todos os documentos fornecidos abaixo
antes de avaliar a solicitação.

Os documentos definem o escopo, as restrições, os requisitos
e as ações permitidas pela aplicação.

Sua tarefa é somente decidir se a solicitação pode continuar.

DOCUMENTAÇÃO OBRIGATÓRIA:

{documentacao}

SOLICITAÇÃO DO USUÁRIO:

{pergunta}

Responda exclusivamente em JSON válido, sem Markdown,
sem texto antes ou depois, usando exatamente esta estrutura:

{{
  "permitido": true,
  "motivo": "motivo técnico e objetivo",
  "resposta": ""
}}

Quando a solicitação estiver proibida ou fora do escopo,
use:

{{
  "permitido": false,
  "motivo": "regra documental que impediu a solicitação",
  "resposta": "resposta curta em português brasileiro explicando a recusa"
}}

Requisitos obrigatórios:

1. Não responda à pergunta principal.
2. Apenas valide a solicitação contra os documentos.
3. Se estiver fora do escopo, permitido deve ser false.
4. Se houver dúvida ou conflito documental, permitido deve ser false.
5. Não autorize uma solicitação que contrarie qualquer documento.
"""
    )

    chain_validacao = (
        prompt_validacao
        | llm
        | StrOutputParser()
    )

    resposta_modelo = chain_validacao.invoke(
        {
            "documentacao": documentacao_obrigatoria,
            "pergunta": pergunta
        }
    )

    try:
        decisao = extrair_json_modelo(
            resposta_modelo
        )
    except Exception as erro:
        print(
            "ERRO ao interpretar validação documental: "
            f"{erro}"
        )

        return {
            "permitido": False,
            "motivo": (
                "Não foi possível interpretar a validação "
                "da documentação."
            ),
            "resposta": (
                "Não posso processar a solicitação porque "
                "a validação das regras documentais falhou."
            ),
            "source": "documentation_validation_error"
        }

    permitido = decisao.get(
        "permitido",
        False
    )

    # Evita que strings como "false" sejam interpretadas
    # como verdadeiras pelo Python.
    if isinstance(permitido, str):
        permitido = permitido.strip().lower() == "true"

    decisao_final = {
        "permitido": permitido is True,
        "motivo": str(
            decisao.get(
                "motivo",
                "Motivo não informado."
            )
        ),
        "resposta": str(
            decisao.get(
                "resposta",
                ""
            )
        ),
        "source": "documentation_validation_llm",
        "documentation_hash": documentacao_hash
    }

    redis_client.setex(
        chave_cache,
        POLICY_CACHE_TTL_SECONDS,
        json.dumps(
            decisao_final,
            ensure_ascii=False
        )
    )

    return decisao_final


# ============================================================
# Indexação RAG
# ============================================================

def carregar_vetores():
    global global_vectorstore

    if not documentos_carregados:
        print(
            "ERRO: Não existem documentos carregados "
            "para indexação."
        )

        global_vectorstore = None
        return False

    try:
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        partes = splitter.split_documents(
            documentos_carregados
        )

        global_vectorstore = RedisVectorStore.from_documents(
            partes,
            embeddings,
            redis_url=REDIS_URL,
            index_name="rag_docs"
        )

        print(
            "SUCESSO: Documentos indexados "
            "no Redis Vector Store."
        )

        return True

    except Exception as erro:
        print(
            f"ERRO ao indexar documentos: {erro}"
        )

        global_vectorstore = None
        return False


# ============================================================
# Detecção MCP
# ============================================================

MCP_KEYWORDS = {
    "cpu",
    "processador",
    "carga",
    "ram",
    "memoria",
    "memory",
    "gpu",
    "placa de video",
    "recurso",
    "recursos",
    "uso da maquina",
    "uso do computador",
    "uso do ubuntu",
    "status da maquina",
    "status do computador",
    "status do ubuntu",
    "telemetria"
}


def consulta_requer_mcp(
    pergunta: str
) -> bool:
    texto = normalizar_texto(
        pergunta
    )

    return any(
        termo in texto
        for termo in MCP_KEYWORDS
    )


def consultar_mcp_metrics() -> dict:
    url = (
        f"{MCP_SERVER_URL}"
        f"{MCP_METRICS_ENDPOINT}"
    )

    try:
        with urlopen(
            url,
            timeout=MCP_TIMEOUT_SECONDS
        ) as response:
            status_code = response.getcode()
            conteudo = response.read().decode(
                "utf-8"
            )

        if status_code != 200:
            raise RuntimeError(
                f"MCP retornou HTTP {status_code}."
            )

        dados = json.loads(
            conteudo
        )

        if dados.get("status") != "online":
            raise RuntimeError(
                "O MCP não está online."
            )

        return dados

    except HTTPError as erro:
        raise RuntimeError(
            f"Erro HTTP do MCP: {erro.code}"
        ) from erro

    except URLError as erro:
        raise RuntimeError(
            f"Erro de conexão com o MCP: {erro.reason}"
        ) from erro

    except json.JSONDecodeError as erro:
        raise RuntimeError(
            "O MCP retornou um JSON inválido."
        ) from erro


def montar_resposta_mcp(
    dados: dict,
    pergunta: str
) -> str:
    texto = normalizar_texto(
        pergunta
    )

    quer_cpu = any(
        termo in texto
        for termo in [
            "cpu",
            "processador",
            "carga"
        ]
    )

    quer_memoria = any(
        termo in texto
        for termo in [
            "ram",
            "memoria",
            "memory"
        ]
    )

    quer_gpu = any(
        termo in texto
        for termo in [
            "gpu",
            "placa de video"
        ]
    )

    consulta_geral = not (
        quer_cpu
        or quer_memoria
        or quer_gpu
    )

    linhas = []

    if quer_cpu or consulta_geral:
        linhas.append(
            "Uso atual de CPU: "
            f"{dados.get('cpu_load', 'não informado')}"
        )

    if quer_memoria or consulta_geral:
        total = dados.get(
            "memory_total_gb"
        )

        usada = dados.get(
            "memory_used_gb"
        )

        if total is not None and usada is not None:
            total = float(total)
            usada = float(usada)
            disponivel = max(
                total - usada,
                0
            )

            percentual = (
                (usada / total) * 100
                if total > 0
                else 0
            )

            linhas.extend(
                [
                    f"Memória RAM utilizada: {usada:.2f} GB",
                    f"Memória RAM total: {total:.2f} GB",
                    (
                        "Memória RAM disponível: "
                        f"{disponivel:.2f} GB"
                    ),
                    (
                        "Percentual de RAM em uso: "
                        f"{percentual:.2f}%"
                    )
                ]
            )
        else:
            linhas.append(
                "O MCP não informou os dados de memória."
            )

    if quer_gpu or consulta_geral:
        gpu = dados.get(
            "gpu_info",
            []
        )

        if gpu:
            linhas.append(
                "Informações de GPU: "
                + json.dumps(
                    gpu,
                    ensure_ascii=False
                )
            )
        else:
            linhas.append(
                "GPU: nenhuma informação disponível."
            )

    timestamp = dados.get(
        "timestamp"
    )

    if timestamp:
        linhas.append(
            f"Timestamp informado pelo MCP: {timestamp}"
        )

    return "\n".join(
        linhas
    )


# ============================================================
# Cache semântico apenas para respostas RAG/LLM
# ============================================================

def buscar_cache_semantico(
    pergunta: str,
    threshold: float = SEMANTIC_CACHE_THRESHOLD
):
    try:
        vetor_atual = embeddings.embed_query(
            pergunta
        )

        melhor_similaridade = 0.0
        melhor_resposta = None
        melhor_chave = None

        for chave in redis_client.scan_iter(
            match="sem_cache:rag:*",
            count=100
        ):
            conteudo = redis_client.get(
                chave
            )

            if not conteudo:
                continue

            try:
                dados = json.loads(
                    conteudo
                )

                # Respostas geradas com documentação antiga
                # não devem ser reutilizadas.
                if (
                    dados.get("documentation_hash")
                    != documentacao_hash
                ):
                    continue

                vetor_salvo = dados.get(
                    "embedding"
                )

                resposta = dados.get(
                    "resposta"
                )

                if not vetor_salvo or not resposta:
                    continue

                similaridade = cosine_similarity(
                    vetor_atual,
                    vetor_salvo
                )

                if similaridade > melhor_similaridade:
                    melhor_similaridade = similaridade
                    melhor_resposta = resposta
                    melhor_chave = chave

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError
            ):
                continue

        if (
            melhor_resposta is not None
            and melhor_similaridade >= threshold
        ):
            return (
                melhor_resposta,
                melhor_similaridade,
                melhor_chave
            )

    except Exception as erro:
        print(
            f"ERRO ao consultar cache semântico: {erro}"
        )

    return None, 0.0, None


def salvar_cache_semantico(
    pergunta: str,
    resposta: str
) -> str:
    embedding = embeddings.embed_query(
        pergunta
    )

    chave = (
        f"sem_cache:rag:"
        f"{int(time.time() * 1000)}"
    )

    conteudo = {
        "pergunta_original": pergunta,
        "embedding": embedding,
        "resposta": resposta,
        "documentation_hash": documentacao_hash,
        "created_at": time.time()
    }

    redis_client.setex(
        chave,
        RAG_CACHE_TTL_SECONDS,
        json.dumps(
            conteudo,
            ensure_ascii=False
        )
    )

    return chave


# ============================================================
# Inicialização
# ============================================================

@app.on_event("startup")
def startup_event():
    try:
        redis_client.ping()
        print("SUCESSO: Redis conectado.")
    except Exception as erro:
        print(
            f"ERRO ao conectar no Redis: {erro}"
        )

    documentacao_ok = carregar_documentacao()

    if documentacao_ok:
        carregar_vetores()
    else:
        print(
            "ERRO: API iniciada em modo restrito porque "
            "a documentação não foi carregada."
        )


# ============================================================
# Endpoints auxiliares
# ============================================================

@app.get("/health")
def health():
    resultado = {
        "api": "online",
        "redis": "offline",
        "mcp": "offline",
        "documentacao_carregada": bool(
            documentacao_obrigatoria
        ),
        "documentos_md": len(
            documentos_carregados
        ),
        "documentation_hash": documentacao_hash,
        "vectorstore_ativo": (
            global_vectorstore is not None
        )
    }

    try:
        redis_client.ping()
        resultado["redis"] = "online"
    except Exception as erro:
        resultado["redis_error"] = str(
            erro
        )

    try:
        dados = consultar_mcp_metrics()
        resultado["mcp"] = dados.get(
            "status",
            "online"
        )
    except Exception as erro:
        resultado["mcp_error"] = str(
            erro
        )

    return resultado


@app.get("/documentacao")
def status_documentacao():
    return {
        "carregada": bool(
            documentacao_obrigatoria
        ),
        "quantidade_documentos": len(
            documentos_carregados
        ),
        "documentation_hash": documentacao_hash,
        "arquivos": [
            documento.metadata.get(
                "file"
            )
            for documento in documentos_carregados
        ]
    }


@app.post("/reindexar")
def reindexar():
    documentacao_ok = carregar_documentacao()

    if not documentacao_ok:
        raise HTTPException(
            status_code=500,
            detail=(
                "Não foi possível carregar os documentos."
            )
        )

    vetores_ok = carregar_vetores()

    return {
        "status": (
            "sucesso"
            if vetores_ok
            else "falha"
        ),
        "documentos_md": len(
            documentos_carregados
        ),
        "documentation_hash": documentacao_hash,
        "vectorstore_ativo": (
            global_vectorstore is not None
        )
    }


# ============================================================
# Endpoint principal
# ============================================================

@app.post("/perguntar")
def perguntar(
    req: QueryRequest,
    request: Request
):
    inicio = time.time()

    pergunta = req.question.strip()

    if not pergunta:
        raise HTTPException(
            status_code=400,
            detail="A pergunta não pode ficar vazia."
        )

    query_text = (
        req.manual_prompt.strip()
        if req.manual_prompt
        else pergunta
    )

    headers = dict(
        request.headers
    )

    header_size_bytes = sum(
        len(str(chave)) + len(str(valor))
        for chave, valor in headers.items()
    )

    # ========================================================
    # ETAPA 1: documentação obrigatória
    # ========================================================

    validacao = validar_documentacao(
        pergunta
    )

    if not validacao.get(
        "permitido",
        False
    ):
        tempo = round(
            time.time() - inicio,
            4
        )

        resposta = validacao.get(
            "resposta"
        ) or (
            "A solicitação foi recusada pelas regras "
            "da documentação obrigatória."
        )

        registrar_metrica(
            source="documentation_guardrail",
            metric_type="documentation_denied",
            question=pergunta,
            context_used=validacao.get(
                "motivo",
                ""
            ),
            header_size_bytes=header_size_bytes,
            execution_time_seconds=tempo,
            extra={
                "documentation_hash": documentacao_hash,
                "validation_source": validacao.get(
                    "source"
                )
            }
        )

        return {
            "source": "documentation_guardrail",
            "execution_time_seconds": tempo,
            "header_size_bytes": header_size_bytes,
            "documentation_validation": {
                "permitido": False,
                "motivo": validacao.get(
                    "motivo"
                )
            },
            "answer": resposta
        }

    # ========================================================
    # ETAPA 2: consulta MCP autorizada
    # ========================================================

    if consulta_requer_mcp(
        pergunta
    ):
        try:
            dados_mcp = consultar_mcp_metrics()

            resposta = montar_resposta_mcp(
                dados=dados_mcp,
                pergunta=pergunta
            )

            tempo = round(
                time.time() - inicio,
                4
            )

            registrar_metrica(
                source="mcp_server",
                metric_type="mcp_query_live",
                question=pergunta,
                context_used=json.dumps(
                    dados_mcp,
                    ensure_ascii=False
                ),
                header_size_bytes=header_size_bytes,
                execution_time_seconds=tempo,
                extra={
                    "documentation_hash": documentacao_hash,
                    "documentation_validation": (
                        "allowed"
                    )
                }
            )

            return {
                "source": "mcp_server",
                "execution_time_seconds": tempo,
                "header_size_bytes": header_size_bytes,
                "documentation_validation": {
                    "permitido": True,
                    "motivo": validacao.get(
                        "motivo"
                    )
                },
                "answer": resposta
            }

        except Exception as erro:
            raise HTTPException(
                status_code=503,
                detail=(
                    "A consulta foi autorizada pela "
                    "documentação, mas o MCP falhou: "
                    f"{erro}"
                )
            )

    # ========================================================
    # ETAPA 3: cache semântico RAG/LLM autorizado
    # ========================================================

    resposta_cache, similaridade, chave_cache = (
        buscar_cache_semantico(
            pergunta
        )
    )

    if resposta_cache:
        tempo = round(
            time.time() - inicio,
            4
        )

        registrar_metrica(
            source="cache_semantico_redis",
            metric_type="rag_query_cache",
            question=pergunta,
            context_used=query_text,
            header_size_bytes=header_size_bytes,
            execution_time_seconds=tempo,
            similarity_score=similaridade,
            extra={
                "cache_key": chave_cache,
                "documentation_hash": documentacao_hash,
                "documentation_validation": "allowed"
            }
        )

        return {
            "source": "cache_semantico_redis",
            "similarity_score": round(
                float(similaridade),
                4
            ),
            "execution_time_seconds": tempo,
            "header_size_bytes": header_size_bytes,
            "documentation_validation": {
                "permitido": True,
                "motivo": validacao.get(
                    "motivo"
                )
            },
            "answer": resposta_cache
        }

    # ========================================================
    # ETAPA 4: RAG + LLM autorizado
    # ========================================================

    try:
        if global_vectorstore is None:
            raise RuntimeError(
                "Vectorstore não inicializado."
            )

        retriever = global_vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

        prompt_rag = ChatPromptTemplate.from_template(
            """
Você é um assistente técnico.

A solicitação já foi previamente autorizada pelo mecanismo
de validação documental.

Mesmo assim, respeite integralmente toda a documentação
obrigatória abaixo.

DOCUMENTAÇÃO OBRIGATÓRIA:

{documentacao}

CONTEXTO RECUPERADO PELO RAG:

{context}

PERGUNTA:

{question}

Responda em português brasileiro, de forma técnica e objetiva.

Não ignore nenhuma restrição da documentação.
Não responda fora do escopo definido pela documentação.
"""
        )

        chain = (
            {
                "documentacao": (
                    lambda _: documentacao_obrigatoria
                ),
                "context": retriever
                | (
                    lambda docs: "\n\n".join(
                        documento.page_content
                        for documento in docs
                    )
                    if docs
                    else "Nenhum contexto adicional encontrado."
                ),
                "question": RunnablePassthrough()
            }
            | prompt_rag
            | llm
            | StrOutputParser()
        )

        resposta = chain.invoke(
            query_text
        )

        chave_cache = salvar_cache_semantico(
            pergunta=pergunta,
            resposta=resposta
        )

        tempo = round(
            time.time() - inicio,
            4
        )

        registrar_metrica(
            source="llm_generated_rag",
            metric_type="rag_query",
            question=pergunta,
            context_used=query_text,
            header_size_bytes=header_size_bytes,
            execution_time_seconds=tempo,
            extra={
                "cache_key": chave_cache,
                "documentation_hash": documentacao_hash,
                "documentation_validation": "allowed"
            }
        )

        return {
            "source": "llm_generated_rag",
            "execution_time_seconds": tempo,
            "header_size_bytes": header_size_bytes,
            "documentation_validation": {
                "permitido": True,
                "motivo": validacao.get(
                    "motivo"
                )
            },
            "answer": resposta
        }

    except Exception as erro:
        print(
            f"ERRO no fluxo RAG: {erro}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "A solicitação foi autorizada, mas não foi "
                f"possível gerar a resposta: {erro}"
            )
        )


# ============================================================
# Métricas
# ============================================================

@app.get("/metricas-laboratorio")
def obter_metricas():
    registros = redis_client.lrange(
        "lab_metrics",
        0,
        -1
    )

    metricas = []

    for registro in registros:
        try:
            metricas.append(
                json.loads(registro)
            )
        except json.JSONDecodeError:
            continue

    total = len(
        metricas
    )

    total_llm = sum(
        1
        for metrica in metricas
        if "llm" in metrica.get(
            "source",
            ""
        )
    )

    total_cache = sum(
        1
        for metrica in metricas
        if "cache" in metrica.get(
            "source",
            ""
        )
    )

    total_mcp = sum(
        1
        for metrica in metricas
        if metrica.get("source") == "mcp_server"
    )

    total_bloqueadas = sum(
        1
        for metrica in metricas
        if metrica.get("source")
        == "documentation_guardrail"
    )

    tempo_medio = (
        sum(
            float(
                metrica.get(
                    "execution_time_seconds",
                    0
                )
            )
            for metrica in metricas
        ) / total
        if total > 0
        else 0
    )

    return {
        "resumo_geral": {
            "total_registros_laboratorio": total,
            "consultas_processadas_llm": total_llm,
            "consultas_reaproveitadas_cache": total_cache,
            "consultas_mcp": total_mcp,
            "consultas_bloqueadas_documentacao": (
                total_bloqueadas
            ),
            "tempo_medio_execucao_segundos": round(
                tempo_medio,
                4
            )
        },
        "documentacao": {
            "carregada": bool(
                documentacao_obrigatoria
            ),
            "arquivos_md": len(
                documentos_carregados
            ),
            "documentation_hash": documentacao_hash
        },
        "detalhado_por_requisicao": metricas
    }
