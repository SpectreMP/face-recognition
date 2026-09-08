# -*- coding: utf-8 -*-
import logging
import os

from flask import Flask, jsonify, render_template_string, request, send_from_directory

from face_store import IMAGE_EXTS, sanitize_name
from voice_cache import format_template

log = logging.getLogger(__name__)

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель управления — распознавание лиц</title>
<style>
  :root { --bg:#12141a; --card:#1b1f2a; --line:#2a3040; --text:#e8eaf0; --muted:#8a93a6;
          --accent:#4f8cff; --ok:#37c26e; --warn:#e6b23c; --err:#e05252; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:'Segoe UI', system-ui, sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:24px 16px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:18px; margin-bottom:18px; }
  label { display:block; font-size:13px; color:var(--muted); margin:10px 0 4px; }
  input[type=text] { width:100%; padding:10px 12px; border-radius:8px; border:1px solid var(--line);
                     background:#10131b; color:var(--text); font-size:15px; }
  input[type=file] { margin-top:6px; color:var(--muted); }
  button { background:var(--accent); color:#fff; border:none; border-radius:8px;
           padding:10px 18px; font-size:14px; cursor:pointer; margin-top:14px; }
  button:hover { filter:brightness(1.1); }
  button.ghost { background:transparent; border:1px solid var(--line); color:var(--muted); padding:5px 10px; margin:0; font-size:12px; }
  button.danger { background:transparent; border:1px solid var(--err); color:var(--err); padding:5px 10px; margin:0; font-size:12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:14px; }
  .person { background:#141824; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .person img { width:100%; height:150px; object-fit:cover; display:block; background:#0d1017; }
  .pbody { padding:10px 12px; }
  .pname { font-weight:600; font-size:15px; word-break:break-word; }
  .pmeta { font-size:12px; color:var(--muted); margin-top:2px; }
  .badge { display:inline-block; font-size:11px; padding:2px 8px; border-radius:20px; margin-top:8px; }
  .b-ready { background:rgba(55,194,110,.15); color:var(--ok); }
  .b-pending { background:rgba(230,178,60,.15); color:var(--warn); }
  .b-missing { background:rgba(138,147,166,.15); color:var(--muted); }
  .b-error { background:rgba(224,82,82,.15); color:var(--err); }
  .pactions { display:flex; gap:6px; padding:0 12px 12px; }
  #log { font-family:Consolas,monospace; font-size:12px; max-height:180px; overflow-y:auto; }
  #log div { padding:3px 0; border-bottom:1px dashed var(--line); }
  .empty { color:var(--muted); text-align:center; padding:30px 0; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Распознавание лиц — панель управления</h1>
  <div class="sub">Добавьте человека: имя + одна или несколько фотографий. Фраза приветствия синтезируется автоматически.</div>

  <div class="card">
    <label for="name">Имя человека</label>
    <input type="text" id="name" placeholder="Например: Иван Петров" maxlength="64">
    <label for="photos">Фотографии (лицо крупно, можно несколько ракурсов)</label>
    <input type="file" id="photos" accept="image/jpeg,image/png" multiple>
    <button onclick="upload()">Добавить</button>
  </div>

  <div class="card">
    <div style="font-weight:600; margin-bottom:10px;">Известные лица</div>
    <div id="persons" class="grid"></div>
    <div id="empty" class="empty" style="display:none">Пока никого нет</div>
  </div>

  <div class="card">
    <div style="font-weight:600; margin-bottom:10px;">Журнал</div>
    <div id="log"></div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
function log(msg, cls) {
  const d = document.createElement('div');
  d.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  if (cls) d.style.color = cls;
  $('#log').prepend(d);
}
async function jfetch(url, opts) {
  const r = await fetch(url, opts);
  let body = null;
  try { body = await r.json(); } catch (e) {}
  return { ok: r.ok, status: r.status, body };
}
const badge = st => ({
  ready:  ['b-ready','озвучка готова'],
  pending:['b-pending','синтезируется…'],
  missing:['b-missing','нет озвучки'],
  error:  ['b-error','ошибка озвучки']
}[st] || ['b-missing', st]);

async function refresh() {
  const res = await jfetch('/api/persons');
  if (!res.ok) return;
  const list = res.body.persons || [];
  $('#empty').style.display = list.length ? 'none' : 'block';
  $('#persons').innerHTML = list.map(p => {
    const [bc, bt] = badge(p.voice_status);
    const photo = p.photos.length ? '/photo/' + encodeURIComponent(p.photos[0]) : '';
    return `<div class="person">
      ${photo ? `<img src="${photo}" alt="">` : '<img alt="">'}
      <div class="pbody">
        <div class="pname">${esc(p.name)}</div>
        <div class="pmeta">фото: ${p.photos.length}</div>
        <span class="badge ${bc}">${bt}</span>
      </div>
      <div class="pactions">
        <button class="ghost" onclick="resynth('${escAttr(p.name)}')">переозвучить</button>
        <button class="danger" onclick="del('${escAttr(p.name)}')">удалить</button>
      </div>
    </div>`;
  }).join('');
}
function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function escAttr(s){ return esc(s).replace(/'/g,'&#39;').replace(/"/g,'&quot;'); }

async function upload() {
  const name = $('#name').value.trim();
  const files = $('#photos').files;
  if (!name) { log('Укажите имя', 'var(--err)'); return; }
  if (!files.length) { log('Выберите хотя бы одно фото', 'var(--err)'); return; }
  const fd = new FormData();
  fd.append('name', name);
  for (const f of files) fd.append('photos', f, f.name);
  log(`Загрузка «${name}» (${files.length} фото)…`);
  try {
    const res = await jfetch('/api/persons', { method:'POST', body: fd });
    (res.body && res.body.results || []).forEach(r =>
      log((r.ok ? '✓ ' : '✗ ') + r.message, r.ok ? 'var(--ok)' : 'var(--err)'));
    if (res.body && res.body.voice)
      log('Озвучка: ' + res.body.voice);
  } catch(e) { log('Ошибка сети: ' + e, 'var(--err)'); }
  $('#name').value = ''; $('#photos').value = '';
  refresh();
}
async function del(name) {
  if (!confirm('Удалить «' + name + '»?')) return;
  const res = await jfetch('/api/persons/' + encodeURIComponent(name), { method:'DELETE' });
  log(res.ok ? `Удалён: ${name}` : `Не удалось удалить: ${name}`, res.ok ? 'var(--ok)' : 'var(--err)');
  refresh();
}
async function resynth(name) {
  const res = await jfetch('/api/persons/' + encodeURIComponent(name) + '/resynthesize', { method:'POST' });
  log(`Переозвучка «${name}»: ${(res.body && res.body.voice) || res.status}`);
  refresh();
}
refresh();
setInterval(refresh, 4000);
</script>
</body>
</html>"""


def create_app(store, voice, faces_dir):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/api/persons")
    def api_persons():
        persons = []
        for name in store.names():
            persons.append({
                "name": name,
                "photos": store.photos(name),
                "voice_status": voice.status(name),
            })
        return jsonify({"persons": persons})

    @app.post("/api/persons")
    def api_add():
        name = request.form.get("name", "")
        files = request.files.getlist("photos")
        if not sanitize_name(name):
            return jsonify({"ok": False, "error": "Некорректное имя", "results": []}), 400
        if not files:
            return jsonify({"ok": False, "error": "Файлы не переданы", "results": []}), 400
        results = []
        added_any = False
        for f in files:
            data = f.read()
            ok, message = store.add_photo(name, data)
            results.append({"ok": ok, "message": message,
                            "filename": os.path.basename(f.filename or "")})
            added_any = added_any or ok
        voice_status = None
        if added_any:
            person_key = sanitize_name(name)
            voice.ensure(person_key)
            voice_status = voice.status(person_key)
        http_code = 200 if any(r["ok"] for r in results) else 422
        return jsonify({"ok": any(r["ok"] for r in results),
                        "results": results, "voice": voice_status}), http_code

    @app.delete("/api/persons/<path:name>")
    def api_delete(name):
        removed = store.delete_person(name)
        if not removed:
            return jsonify({"ok": False, "error": "Человек не найден"}), 404
        voice.invalidate(name)
        return jsonify({"ok": True})

    @app.post("/api/persons/<path:name>/resynthesize")
    def api_resynth(name):
        if not store.has(name):
            return jsonify({"ok": False, "error": "Человек не найден"}), 404
        text = format_template(voice.template, name)
        voice.ensure(name, force=True)
        return jsonify({"ok": True, "voice": voice.status(name), "phrase": text})

    @app.get("/photo/<path:fname>")
    def photo(fname):
        if not fname.lower().endswith(IMAGE_EXTS):
            return jsonify({"ok": False, "error": "Недопустимый тип файла"}), 404
        return send_from_directory(store.faces_dir, fname)

    return app
