# Skill Móvil · Visión y Desambiguación de Etiquetas Patrimoniales

## Objetivo
Extraer códigos patrimoniales desde etiquetas dañadas, borrosas, con reflejo o escritas a mano en campo.

## Reglas de Extracción
1. Descartar palabras institucionales no identificadoras: `UAGRM`, `ACTIVO`, `FIJO`, `PATRIMONIO`, `BOLIVIA`, `FICCT`, `DAF`, `SERIE`, `MODELO`, `MARCA`.
2. Conservar únicamente tokens alfanuméricos de 4 a 24 caracteres con al menos 2 dígitos.
3. Unir pares fragmentados consecutivos cuando forman un código continuo (ej. `U01 000045` -> `U01000045`).
4. Generar variante correctiva de confusión visual frecuente (`O->0`, `I/L->1`, `S->5`, `B->8`, `Z->2`) en segmentos numéricos cuando exista alta densidad de dígitos o coincida con prefijos conocidos del catálogo, manteniendo el candidato literal primero.
5. Escalar pasadas de contraste (`original` -> `clahe_realce` -> `umbral_adaptativo`) con salida temprana cuando la confianza supere `0.78`.
