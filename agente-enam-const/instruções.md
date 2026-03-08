Crie um agente com base em RAG e busca vetorial, com api da Claude, que justifica cada questão do arquivo csv
verificando os chunks da constituição, dos informativos tematicos de 2026, 2025 e 2024, da doutrina e da jurisprudencia de repercussao geral.

Primeiro busque na constituição, depois na repercussão geral, depois nas súmulas vinculantes, depois na doutrina (pdf DIREITO CONSTITUCIONAL) e por fim nos informativos. Não use informação da internet e seu conhecimento sem estar nos documentos do RAG.
não gere uma interface, apenas rode no terminal e salve um csv a cada 10 questões atualizadas.

Gere um quero o mesmo csv (questoes extraidas - CONSTITUCIONAL) com a adição de uma uma coluna com a explicação detalhada a partir dos chunks. Os dados devem ser persistidos em um banco de dados.

No system prompt do agente use o seguinte one shot para melhor a forma como a explicação será dada:
A alternativa correta é a letra C. De acordo com art. 17, § 8º da CF: “§ 8º O montante do Fundo Especial de Financiamento de Campanha e da parcela do fundo partidário destinada a campanhas eleitorais, bem como o tempo de propaganda gratuita no rádio e na televisão a ser distribuído pelos partidos às respectivas candidatas, deverão ser de no mínimo 30% (trinta por cento), proporcional ao número de candidatas, e a distribuição deverá ser realizada conforme critérios definidos pelos respectivos órgãos de direção e pelas normas estatutárias, considerados a autonomia e o interesse partidário.”

Explique somente o motivo e a justificativa fundamenta da questão correta. Não responda o motivo das demais questões estarem erradas.

Crie um .env com a seguinte chave Claude: CHAVE_REMOVIDA_DO_HISTORICO
