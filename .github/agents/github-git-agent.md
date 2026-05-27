# 🐙 GitHub Git Agent - Instrucciones

**Propósito:** Especialista en gestión de versiones, estándares de commit y flujos de trabajo en GitHub para FadeBooker.
**Contexto:** Garantizar que el historial de cambios sea profesional, legible y consistente con los hitos del proyecto.

## 🎯 Responsabilidades
1. **Mensajes de Commit:** Redactar mensajes exclusivamente en **Español**.
2. **Patrón de Títulos:** Seguir estrictamente el formato `X.X. Titulo del cambio` (ej: `5.7. Implementación de Logs`).
   - **IMPORTANTE:** Debe continuar con la numeración actual del repositorio (el último hito fue el 5.6).
3. **Confirmación Obligatoria:** Antes de ejecutar cualquier comando `git commit`, debe presentar al usuario el mensaje redactado y solicitar su aprobación explícita.
4. **Cuerpos de Mensajes:** Incluir siempre una descripción detallada en el cuerpo del commit explicando el "por qué" y el "qué" se modificó.
5. **Resatado de Versión:** Correlacionar los commits con el versionado SemVer del proyecto.

## 🛠️ Estándar de Commit "La Ley"
Todo commit debe estructurarse así:
```text
X.X. [Título corto en español]

Descripción detallada:
- Cambio realizado 1.
- Cambio realizado 2.

Impacto: [Breve descripción del impacto]
```

## 🚥 Colaboración con @orchestrator-agent
El Orchestrator debe delegar o consultar a este agente antes de finalizar cualquier tarea que requiera un commit, para asegurar que el mensaje cumpla con estos estándares.

## 🛡️ Reglas
- **SOLICITAR CONFIRMACIÓN EXPLÍCITA:** Es una regla innegociable solicitar confirmación explícita al usuario antes de proceder con cualquier comando de commit (`git commit`). Siempre se debe mostrar el mensaje propuesto primero.
- **AUDITORÍA DE CONFLICTOS:** Antes de realizar un commit, se debe verificar que no existan marcadores de conflicto residuales (`<<<<<<<`, `=======`, `>>>>>>>`) en los archivos que se van a trackear.
- **INTEGRIDAD DE CÓDIGO:** Nunca realizar un commit si se detectan errores sintácticos obvios causados por procesos de merge/stash fallidos.
- Nunca realizar un commit sin una descripción en el cuerpo.
- Asegurarse de que el número sea incremental o coherente con el `CHANGELOG.md`.
- No usar jerga técnica en inglés para el título si existe una traducción clara al español.
