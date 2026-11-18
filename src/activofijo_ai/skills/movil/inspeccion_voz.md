Responde SOLO JSON: {"a":"completar|confirmar|cancelar|ignorar","e":"BUENO|REGULAR|MALO|null","o":"observacion tecnica <=8 palabras o vacio"}
Ejemplos:
etapa=dictado_inicial|activo=MESA|dicho=buenos dias como estas -> {"a":"ignorar","e":null,"o":""}
etapa=dictado_inicial|activo=PROYECTOR|dicho=oye pasame mi mochila que nos vamos a almorzar -> {"a":"ignorar","e":null,"o":""}
etapa=dictado_inicial|activo=SILLA|dicho=hola maria, eh la silla tiene el espaldar medio flojo -> {"a":"completar","e":"REGULAR","o":"Espaldar flojo"}
etapa=dictado_inicial|activo=MESA|dicho=mal estado le falta una pata trasera -> {"a":"completar","e":"MALO","o":"Falta una pata trasera"}
etapa=esperando_confirmacion|activo=PC|dicho=listo jefe ya guardalo asi como esta -> {"a":"confirmar","e":null,"o":""}
etapa=dictado_inicial|activo=SILLA|dicho=mejor no, cancelalo -> {"a":"cancelar","e":null,"o":""}
