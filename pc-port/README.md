# inFAMOUS Second Son — protótipo de runtime para Windows

Atualização vigente: [sessão 07](evidence/mapping/SESSAO-07.md), 124 funções
catalogadas e trace opcional de comandos. Os principais elos das partículas
foram ligados a shaders/draws; a hipótese geral de eliminar barriers entre
dispatches consecutivos não se sustenta. ISS_COMPUTE_PHASES continua experimental
e desligado. Tradução equivalente e 60 FPS estáveis ainda pendentes.

Mapeamento atualizado (07/10, noite): [índice e retomada](evidence/mapping/RETOMADA.md)
e [sessão 06](evidence/mapping/SESSAO-06.md). Inclui 97 funções catalogadas,
identidades de shaders capturadas em gameplay e contratos de sincronização.
O mapa é parcial; tradução completa e 60 FPS estáveis continuam pendentes.

Investigação de desempenho (07/10, tarde): gameplay e salvamento já funcionam;
60 FPS sustentados continuam pendentes. O diagnóstico detalhado e a cadeia de
feedback GPU→CPU recuperada no Ghidra estão em
[`evidence/gpu-feedback-analysis.md`](evidence/gpu-feedback-analysis.md).
`ISS_DEEP_PROFILE=1` adiciona `cpu-phases.csv` e `memory-access.csv` por execução;
`python scripts/summarize_deep_profile.py CAMINHO_DO_RUN` resume os intervalos
finais. O perfil detalhado tem custo e fica desligado por padrão.

Existe uma reconstrução C++ experimental da redução em `native/`, validada
contra a sequência AVX original. `ISS_NATIVE_REDUCTION=1` a ativa somente após
verificação da assinatura em memória. O A/B não mostrou ganho de FPS; ela
permanece desligada. Recompilar uma função isolada não remove suas dependências
de dados da GPU.

Estado em **07/10/2026**: inicializadores nativos de libc/Fios2 concluídos;
abertura, áudio e cenas 3D reais renderizados pelo runtime Windows e `iss-gpu.dll`.
A transição para gameplay revelou falhas Windows 998 nas leituras para memória
protegida pelo renderer. A transferência agora usa um buffer host intermediário
e copia em modo usuário, permitindo que o handler de memória gráfica execute.
O teste nativo de buffers protegidos passou. Na reprodução gráfica, os dois
trechos que falhavam foram lidos integralmente e o jogo chegou ao tutorial de
segurar o controle de lado, sem o aviso. A entrada agora recebe touch e movimento
do DS4 virtual do PhonePad; gyro/aceleração reais foram observados no port.
Ainda não há garantia de estabilidade, fidelidade completa ou 60 FPS.

## Execução gráfica e logs atuais

```powershell
python pc-port/scripts/build.py
python pc-port/scripts/build_gpu.py
python pc-port/scripts/run_probe.py --linked --entry --gpu --timeout 180
```

Compile com o processo do jogo fechado. O prazo solicita gravação dos caches e
encerra o processo inteiro, incluindo áudio (kill é fallback após 15 segundos).
Código de saída 1 do script após timeout é esperado neste teste;
não equivale a um crash nem confirma gameplay correto.

Cada execução cria `out/runs/<data-hora>/` com `run.json`, `stdout.log`,
`stderr.log` e `io.tsv`. O TSV registra tempo monotônico, thread, operação,
descritor, offset, bytes pedidos/transferidos, erro Windows, errno e caminho.
`run.json` guarda hashes, PID, prazo, eventos e resumo das transferências.
`out/latest-run.json` aponta para a execução mais recente. Saves locais anteriores
são copiados para `savedata-before/` de cada execução gráfica. Testes com prazo
usam outra cópia em `savedata-work/`, isolando as gravações do jogo durante benchmarks.
O launcher sem prazo continua usando os saves normais. Não há cópia dos assets.

Capturas em `out/user/gpu-2/` são opcionais (`ISS_CAPTURE_FRAMES=1`, a cada 15
segundos e nos primeiros marcos). Ficam desligadas normalmente para evitar custo
de readback/PNG durante o jogo. Avisos repetidos preservam a primeira ocorrência e contagens
em potências de dois (`GPU_REPEAT`), com tabela limitada a 4096 mensagens.
O diálogo do jogo continua visível e é registrado com tempo, thread e caller.

`frames.csv` registra tempo de apresentação, compilação de shaders/pipelines,
processamento de comandos, readbacks, uploads, barreiras e timestamps de GPU.
`process.csv` amostra a cada segundo CPU do processo (percentual da máquina e
equivalentes de núcleos), memória residente/privada, page faults e bytes de I/O.
F10 escreve `PERF_MARK` no `stderr.log` para localizar uma travada relatada.
Ocultar o contador com F9 não desliga esses logs.

Controles copiados de `sandbox/experimental/user/input_config/CUSA00004.ini`
(janela do jogo focada):

| Entrada | Ação |
|---|---|
| WASD | Movimento |
| I/J/K/L | Câmera |
| Setas | Direcional |
| N / numpad 2 | Cross |
| B / numpad 6 | Circle |
| V / numpad 4 | Square |
| C / numpad 8 | Triangle |
| Q / U | L1 / R1 |
| E / O | L2 / R2 |
| X / M | L3 / R3 |
| Esc | Options / Start do jogo |
| Espaço ou Tab | Contato esquerdo + clique no touchpad |
| Backspace | Contato direito + clique |
| F8 | Dois contatos + clique |
| 1 / 2 / 3 / 4 | Swipe cima / baixo / esquerda / direita (450 ms) |
| Select/Back no controle | Contato esquerdo + clique |
| Start/Options no controle | Contato direito + clique |
| Select + Start | Dois contatos simultâneos + clique |
| F9 | Mostrar/ocultar FPS e ms |
| F10 | Marcar uma travada no log |

O touchpad e os sensores reais do PhonePad continuam disponíveis. Os botões
Start/Select do controle seguem o remapeamento do fork; use Esc para o menu.
O contador usa flips do jogo, sem contar reapresentações do mesmo frame, e mostra
média de FPS/tempo por frame atualizada a cada meio segundo. Não é tempo exclusivo
de GPU nem medida de velocidade da simulação. Saves ficam somente em `out/user/`.

## PhonePad, salvamento manual e desempenho

Abra o PhonePad Companion e ative o DS4 virtual; conecte o celular. O port escolhe
automaticamente PS4/PS5 via SDL antes dos controles Xbox, inclusive se aparecerem
após o boot. O DS4 do Companion já combina botões/analógicos do Xbox com os sensores
do celular. Não é necessário modificar o app PhonePad ou abrir novas portas.
Dois contatos usam coordenadas e IDs independentes; tocar não força o clique.
Gyro chega em rad/s, aceleração em g, e o quaternion é integrado do gyro.
A orientação pode acumular drift; o protocolo HID não entrega o quaternion
absoluto exibido na prévia do celular. `pad-test.exe --monitor` registra 30 segundos
da entrada real sem iniciar o jogo; o teste automático usa somente dispositivo virtual.

`src/save_dialog.c` fornece seleção de slots, novo salvamento, mensagens e
confirmações através de janela Windows. Use mouse/teclado nessa janela. O jogo
grava os dados depois que o usuário escolhe; abrir o diálogo não significa salvar.
O crash original era uma parada em `sceSaveDataDialogInitialize` não implementada.
O ciclo de vida/ABI passou nos testes e a gravação manual foi confirmada pelo usuário.
A busca agora interpreta `%` e `_` conforme o PS4, ordena antes de limitar a lista
e devolve datas reais dos metadados. O carregamento automático escolhe o slot mais
recente, inclusive autosaves; use o menu Carregar para escolher um manual anterior.

Cada execução registra `frames.csv`. Analise os últimos 30 segundos com
`python pc-port/scripts/analyze_performance.py`; o resultado fica em
`performance-summary.json` na mesma pasta. `ISS_GPU_PROFILE=1` habilita timestamps
da GPU para investigação; o `Iniciar-PC-Port.cmd` já o habilita. Tempos de CPU/GPU se
sobrepõem; não devem ser somados. FPS de apresentação não comprova velocidade da
simulação. A amostra de 07/10 incluiu abertura/menu/transições e não é um benchmark
de mundo aberto. Readbacks e processamento de comandos merecem investigação.

O runner ativa pré-carga paralela de pipelines conhecidos. O cache do driver em
`out/user/gpu-14/CUSA00004.vkcache` é persistido por checkpoint atômico a cada 30
segundos e ao fechar a janela/Ctrl+C/prazo. Uma interrupção abrupta ainda pode
perder alterações posteriores ao último checkpoint. O arquivo usa a validação
de dispositivo/UUID/checksum existente no renderer. `ISS_PIPELINE_CHECKPOINT=0`
desliga checkpoints para comparação e `SHADPS4_PARALLEL_PRELOAD=0` desliga a
pré-carga paralela. O modo que pula draws durante compilação continua desligado.

`ISS_BULK_COPY=0` permite comparar com a cópia antiga em blocos de 16 KB; o padrão
copia regiões integralmente mapeadas de uma vez, mantendo zero-fill de lacunas.
`ISS_UPLOAD_WORKER=1` é experimental e não é habilitado por padrão. O teste
`python pc-port/scripts/benchmark_uploads.py --bulk` compara cópia antiga/nova;
sem `--bulk`, compara worker desligado/ligado. Ambos usam um seed idêntico de saves
isolados e escrevem relatórios em `out/benchmarks/`. Não generalizar uma medição
curta da abertura para desempenho de todo o jogo.

## Histórico do primeiro marco

As seções abaixo documentam o bootstrap anterior à integração gráfica;
suas paradas e limitações de implementação descrevem aquele marco histórico.

O projeto fica em `pc-port/`, no workspace atual. O checkout
`research/bloodborne_pc/` serve como referência de arquitetura e código de leitura
SELF. Não substituímos a build experimental do emulador nem seus launchers.
O alvo inicial é Windows nativo na máquina atual, com um runtime específico para
inFAMOUS. Aqui “port” significa executar o binário original com serviços e tradução
gráfica especializados; não temos o código-fonte original do jogo.

## Rodar este marco

Na raiz do workspace, abra **`Testar-PC-Port.cmd`**. Ele prepara a imagem, compila o
loader/runtime, executa os testes sintéticos, valida o mapeamento e roda o baseline
e a imagem ligada até a primeira chamada não implementada. Não é um launcher para jogar.

Ou execute no PowerShell:

```powershell
python pc-port/scripts/prepare.py
python pc-port/scripts/link_modules.py
python pc-port/scripts/build.py
python -m unittest discover -s pc-port/tests -v
python pc-port/scripts/run_probe.py
python pc-port/scripts/run_probe.py --entry
python pc-port/scripts/run_probe.py --linked
python pc-port/scripts/run_probe.py --linked --entry
```

Os comandos dependem do Python, VS2022 SDK, Clang 19/lld e das bibliotecas Zydis/Zycore
já compiladas em `build-opt`. Esses arquivos também já estão no workspace.
Não exigem instalação nova, Linux ou WSL. O preparador usa por padrão:

```text
--game "<pasta do seu jogo>"   (ou ISS_GAME_DIR, ou a primeira linha de pc-port/out/game_dir.txt)
```

Só lê `eboot.bin`, `sce_sys/param.sfo` e os dois `.prx` em `sce_module`.
Não copia o conjunto de assets. Todo arquivo gerado fica em `pc-port/out/`.
Esse diretório está ignorado pelo Git e contém dados derivados do dump local.

## Resultado observado

| Item | Resultado |
|---|---|
| Jogo | CUSA00004 (APP_VER 01.00 / VERSION 01.07) ou CUSA00223 (APP_VER 01.07 / VERSION 01.00); mesmo eboot |
| SHA256 eboot | `2d1ca79630d7bbe6fa29575f59aa7996d43071d74ac8e4f7d40137967d9039ab` |
| Imagem principal | 133.846.143 bytes, com BSS zerado |
| Entry point | offset `+0x61dfd0` |
| Relocations aplicadas | 96.139 |
| Imports principais usados nas relocations | 466, em 26 bibliotecas |
| Módulos declarados pelo eboot | 28 |
| Imports gráficos | 85 GnmDriver + 13 VideoOut |
| Imports com candidato exato nos PRX locais | 142; ainda não vinculados |
| Testes sintéticos do loader | 15 passaram |
| Mapeamento do eboot real | passou, sem executar guest |
| Execução inicial do eboot real | parou em `bzQExy189ZI#c#c`, NID de `_init_env` |
| Endereço de retorno observado | `+0x61dfe9`, correspondente ao primeiro CALL da entrada |

A execução real carregou a imagem em `0x10000000`; isso é um endereço escolhido
naquela execução, não uma identidade do programa. Endereços de investigação devem
usar offsets relativos. A importação usa identidade NID + biblioteca + versão +
módulo + versão, não o sufixo local observado no dump de Bloodborne.

`out/probe-entry.json` registra o hash do loader e da imagem, o exit code, a
exceção, o import e o ponto de retorno. O processo filho saiu com código **20**,
que neste protótipo significa parada esperada por import não implementado.
O script retorna sucesso quando identifica essa parada; isso não significa que
a inicialização do jogo tenha sido concluída.

## Avanço da imagem ligada

`out/boot-linked.bin` (ISSPRB02) inclui o eboot e ambos os PRX do dump local,
sem alterar instruções dos originais ou aplicar patches de Bloodborne.
Tem 136.545.613 bytes, 9 segmentos, 98.123 relocations a aplicar no loader,
564 identidades importadas e 147 bindings nativos por identidade/versões exatas.
As 16 relocations `R_X86_64_DTPMOD64` dos PRX recebem o ID TLS do módulo dono.
Não inferimos aliases de `libSceLibcInternal` para libc: divergências ficam visíveis.

O runtime implementa os contratos necessários até aqui: canário de stack,
`__tls_get_addr`, `sceKernelGetProcParam`, `_sceKernelRtldSetApplicationHeapAPI`
e `sceLibcHeapGetTraceInfo`, com a tabela/bitmask que a libc utiliza.
O último contrato foi conferido com a implementação de shadPS4.
As funções nativas de libc fazem o restante dessas etapas; não usamos retornos
universais de sucesso para imports desconhecidos.

Resultado real da imagem ligada: entra no initializer da libc, registra sua API
de heap e passa por consulta, alocação e mapeamento da memória direta solicitada
por um callback do eboot. A parada seguinte é `scePthreadMutexattrInit`, com retorno
`+0x800d7ec` na imagem ligada (`libc.prx+0x5d7ec`). A libc ainda está inicializando;
Fios2 foi vinculado, mas seu initializer não chegou a executar.
O evento registra `phase=module_init` e `active_module=1`.

No Ghidra, `FUN_012ff6e0` corresponde ao offset `+0x2ff6e0` na importação atual,
cuja base é `0x1000000`. A função consulta o pool físico, solicita uma alocação
com alinhamento de 64 KiB, mapeia-a e cria um heap. A configuração inicial observada
solicita `0x48000000` bytes (1.152 MiB). O bootstrap de memória direta atende essa
alocação usando uma section Windows com `SEC_RESERVE` e views compartilhadas,
sem copiar dados para sincronizar aliases. O pool lógico foi configurado com
4,5 GiB; as páginas são comprometidas conforme os mapeamentos. Isso reserva
backing virtual, não prova consumo de 1.152 MiB de RAM física nem desempenho.

`src/memory.c` implementa consulta de capacidade, alocação de faixas sem sobreposição
e mapeamento, incluindo offsets físicos de 16 KiB dentro de views Windows de 64 KiB.
O teste verifica escrita/leitura entre dois aliases, erro em faixa não alocada e
colisão de endereço fixo. O suporte é de bootstrap CPU: release/unmap/protect pelo
guest, alocação flexível, flags/alignments adicionais e coerência com GPU ainda
precisam de implementação. Tipos de memória são validados, mas ainda não possuem
políticas distintas de cache/coerência.

`evidence/ghidra-startup.json` contém metadados, bytes da entrada e pseudocódigo
obtidos pelo MCP **apenas** do programa `/PS4/eboot-ps4.elf`. A conexão TCP existente
funcionou; a descoberta UDS por nome de projeto falhou. O outro programa aberto
foi ignorado. A importação GUI atual é parcial e genérica; seus bytes de entrada
foram comparados com o eboot conhecido. Nenhuma alteração de image base foi feita.

## Código e evidências

| Arquivo | Função |
|---|---|
| `scripts/ps4_format.py` | leitura SELF/SFO, adaptada da referência |
| `scripts/prepare.py` | validação de identidade, inventário, imagem e relocations |
| `scripts/build.py` | compilação exclusiva de `iss-probe.exe` |
| `src/probe.c` | mapeamento Windows, transição SysV e traps por import |
| `scripts/link_modules.py` | linkagem offline dos PRX e relocations TLS por módulo |
| `src/runtime.c` | TCB/DTV, TLS por thread e contratos de bootstrap |
| `src/memory.c` | faixas físicas e views Windows compartilhadas para memória direta |
| `scripts/run_probe.py` | verificação de hashes, processo filho com prazo de 10 s e relatório |
| `tests/test_probe.py` | imagens sintéticas para ABI, relocations, traps e entradas inválidas |
| `out/analysis.json` | segmentos, TLS, módulos, hashes e resumo de dependências |
| `out/imports.csv` | imports referenciados por relocations do eboot e dos PRX |
| `out/probe-map.json` | resultado do mapeamento real |
| `out/probe-entry.json` | resultado da execução inicial real |
| `out/link.json` / `out/linked-imports.csv` | mapa da imagem ligada e identidades exatas |
| `out/probe-linked-entry.json` | parada real durante inicialização nativa da libc |
| `tests/runtime_test.c` | templates, zero fill, duas threads simultâneas, TCB/DTV e instruções TLS |
| `scripts/validate.py` / `out/validation.json` | reprodução e evidência consolidada dos checks |

O loader usa memória baixa, abaixo de 4 GiB; aplica relocations antes de permitir
execução e publica as permissões dos segmentos antes de entrar no guest.
Imports de função apontam para traps identificáveis. Imports de dados apontam
para páginas inacessíveis: não fornecemos valores inventados. Falhas guest também
produzem diagnóstico e encerramento do filho. O prazo limita somente esse filho.

Os testes atuais usam a stack do thread host. A imagem ligada configura TLS guest
por módulo/thread e mantém FS/GS e o TLS do Windows intactos. Um handler de diagnóstico
emula somente leituras TCB absolutas de 64 bits MOV/XOR/CMP via FS, decodificadas com
Zydis no endereço realmente executado. Ele preserva registradores e flags relevantes
e não modifica código com threads em execução. Esse caminho é deliberadamente lento
e ainda não é uma solução de desempenho para gameplay.

O teste nativo verifica templates, zero fill, isolamento com duas threads vivas
simultaneamente, TCB/DTV, resolução por versão, semântica MOV/XOR/CMP e memória direta.
Os 24 testes
Python cobrem o loader e linkagem, incluindo incompatibilidade/ambiguidade de exports.
A execução real ligada ainda não realizou leituras FS emuladas antes da parada atual;
a validação desse caminho é sintética. Não configuramos stack guest própria,
tratamento completo de exceções ou criação de threads PS4. Logo, **não demonstramos
compatibilidade de todo o código do jogo com Windows**.

Os nomes de APIs no inventário são hints obtidos da AeroLib; não provam execução
em gameplay. Contagens e categorias vêm de imports com relocations, não de strings
soltas. O CSV possui 631 linhas considerando o eboot e ambos os PRX; 630 receberam
hint de nome. O NID `fJgP+wqifno`, de `libSceDiscMap` requerido por Fios2, permanece
sem nome conhecido nesse banco.

O SELF não disponibiliza um header final de metadados de cada módulo. Nenhum deles
é necessário para os segmentos carregáveis/dinâmicos exigidos neste marco; o
relatório lista seus índices. O hash do ELF reconstruído do eboot corresponde ao
ELF reconstruído anterior do workspace, não a uma prova de recuperar metadados ausentes.

A verificação existente de seis fingerprints dos originais passou antes e depois
do trabalho. Ela continua sendo uma verificação desses seis arquivos, não uma
auditoria completa de todas as pastas.

## Próximos marcos

1. **Inicialização libc e módulos:** resolver exports por identidade/versões,
   aplicar as relocations dos PRX, construir TLS por módulo e thread e validar
   ordem de inicialização/finalização. Parar em cada API ausente com evidência.
   libc tem TLS de 416 bytes iniciais / 1.168 bytes totais; Fios2 tem 0 / 64 bytes;
   o eboot tem 4 / 48 bytes. Linkagem e TLS de bootstrap foram implementados;
   concluir construtores, stack própria e ciclo de vida dos módulos continua pendente.
2. **Runtime Windows:** memória, arquivos, threads e sincronização, tempo,
   input e áudio. Conferir ABI, layouts e códigos de erro com o emulador existente.
   Arquivos do jogo continuam na fonte; saves/cache novos terão destino próprio.
3. **Primeiro frame:** integrar uma biblioteca de renderer Vulkan e o shader
   recompiler. A referência tem dependências Linux/POSIX/X11 no runtime e build
   gráfico; não é diretamente compilável como solução Windows completa.
4. **Correção de gameplay:** comparação visual, saves, touchpad de dois contatos,
   áudio, pacing e simulação com velocidade normal.
5. **Desempenho:** avaliar decode e preparação/recording separados, sem dispensar
   readbacks e sem publicar alterações de código com threads em execução.
   Ganhos de Bloodborne não são medições de inFAMOUS; 60 FPS ainda não está validado.

Veja `ARCHITECTURE.md` para o que podemos reaproveitar da referência e
`THIRD_PARTY.md` para origem e licença do código.
