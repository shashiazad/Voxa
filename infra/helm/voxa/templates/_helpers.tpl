{{- define "voxa.image" -}}
{{ .root.Values.image.registry }}/voxa-{{ .name }}:{{ .root.Values.image.tag }}
{{- end -}}
{{- define "voxa.labels" -}}
app.kubernetes.io/name: voxa
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
{{- define "voxa.secretEnv" -}}
{{- $root := .root -}}
{{- range .keys }}
- name: {{ . }}
  valueFrom: { secretKeyRef: { name: {{ $root.Values.secrets.existingSecret }}, key: {{ . }} } }
{{- end }}
{{- end -}}
{{- define "voxa.secCtx" -}}
securityContext:
  runAsNonRoot: {{ .Values.podSecurity.runAsNonRoot }}
  runAsUser: {{ .Values.podSecurity.runAsUser }}
  allowPrivilegeEscalation: false
  capabilities: { drop: [ALL] }
{{- end -}}
