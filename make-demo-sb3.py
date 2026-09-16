#!/usr/bin/env python3
"""Генерирует демо-проект demo-torch.sb3 для расширения Scratch Torch.

Открывать так:  turbowarp.org/editor?extension=<URL>/scratch-torch.js
затем File -> Load from your computer -> demo-torch.sb3
"""
import hashlib
import io
import json
import os
import struct
import zlib
import zipfile

IDS = iter([f"b{i}" for i in range(1, 1000)])

def nid():
    return next(IDS)

def shadow_text(value):
    """[1, [10, value]] — text-примитив, block==shadow (INPUT_SAME_BLOCK_SHADOW)"""
    return [1, [10, value]]

def shadow_num(value):
    """[1, [4, value]] — math_number-примитив"""
    return [1, [4, value]]

def block_ref(block_id):
    """[3, blockId, [10, ""]] — репортёр поверх shadow (INPUT_DIFF_BLOCK_SHADOW)"""
    return [3, block_id, [10, ""]]

def make_block(opcode, inputs, x, y, top=True, nxt=None, parent=None):
    return {
        "opcode": opcode,
        "next": nxt,
        "parent": parent,
        "inputs": inputs,
        "fields": {},
        "shadow": False,
        "topLevel": top,
        "x": x,
        "y": y,
    }

blocks = {}
stack_y = 0

def stack(opcode, inputs, x=40):
    global stack_y
    bid = nid()
    blocks[bid] = make_block(opcode, inputs, x, stack_y, top=False)
    return bid

# ===== скрипт =====
# 1) hat
hat = nid()
blocks[hat] = make_block("event_whenflagclicked", {}, 40, 0, top=True)

# 2) dfCreate
csv = (
    "s1,p1,species\n"
    "5.1,1.4,setosa\n"
    "4.9,1.4,setosa\n"
    "5.0,1.5,setosa\n"
    "5.4,1.5,setosa\n"
    "6.2,2.2,virginica\n"
    "6.0,2.1,virginica\n"
    "6.1,2.2,virginica\n"
    "6.3,2.0,virginica\n"
    "6.0,1.7,versicolor\n"
    "5.9,1.6,versicolor\n"
    "5.8,1.6,versicolor\n"
    "6.1,1.6,versicolor"
)
df_id = stack("scratchtorch_dfCreate", {
    "name": shadow_text("df1"),
    "csv": shadow_text(csv),
    "sep": shadow_text("запятая ,"),
})
blocks[hat]["next"] = df_id
blocks[df_id]["parent"] = hat

# 3) modelCreate: m1 = Sequential(Dense8(relu) -> Dense3(softmax))
dense3 = nid()
blocks[dense3] = make_block("scratchtorch_nnDense", {
    "units": shadow_num(3),
    "act": shadow_text("softmax"),
    "next": shadow_text(""),
}, 40, 0, top=False)

dense8 = nid()
blocks[dense8] = make_block("scratchtorch_nnDense", {
    "units": shadow_num(8),
    "act": shadow_text("relu"),
    "next": block_ref(dense3),
}, 40, 0, top=False)
blocks[dense3]["parent"] = dense8

mc_id = stack("scratchtorch_modelCreate", {
    "name": shadow_text("m1"),
    "first": block_ref(dense8),
})
blocks[df_id]["next"] = mc_id
blocks[mc_id]["parent"] = df_id
blocks[dense8]["parent"] = mc_id

# 4) fit
fit_id = stack("scratchtorch_fit", {
    "model": shadow_text("m1"),
    "table": shadow_text("df1"),
    "features": shadow_text("s1,p1"),
    "label": shadow_text("species"),
    "epochs": shadow_num(150),
    "batch": shadow_num(4),
    "loss": shadow_text("CategoricalCrossentropy"),
    "opt": shadow_text("adam"),
})
blocks[mc_id]["next"] = fit_id
blocks[fit_id]["parent"] = mc_id

# 5) предсказание: say(предскажи m1 для "5.1,1.4")
pred_id = nid()
blocks[pred_id] = make_block("scratchtorch_predict", {
    "model": shadow_text("m1"),
    "input": shadow_text("5.1,1.4"),
}, 40, 0, top=False)

say1 = nid()
blocks[say1] = make_block("looks_say", {
    "MESSAGE": block_ref(pred_id),
}, 40, 0, top=False)
blocks[fit_id]["next"] = say1
blocks[say1]["parent"] = fit_id
blocks[pred_id]["parent"] = say1

# 6) score: say(метрика m1 на df1)
score_id = nid()
blocks[score_id] = make_block("scratchtorch_score", {
    "model": shadow_text("m1"),
    "table": shadow_text("df1"),
    "features": shadow_text("s1,p1"),
    "label": shadow_text("species"),
}, 40, 0, top=False)

say2 = nid()
blocks[say2] = make_block("looks_say", {
    "MESSAGE": block_ref(score_id),
}, 40, 0, top=False)
blocks[say1]["next"] = say2
blocks[say2]["parent"] = say1
blocks[score_id]["parent"] = say2

# фикс: parent у first/nested надо оставить так, как задано (dense8 parent = mc_id)

# ===== костюм: крошечный PNG =====
def tiny_png():
    # 1x1 прозрачный PNG
    def chunk(t, data):
        c = struct.pack(">I", len(data)) + t + data
        c += struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
        return c
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00\x00\x00")
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))

png = tiny_png()
asset_id = hashlib.md5(png).hexdigest()

project = {
    "targets": [
        {
            "isStage": True,
            "name": "Stage",
            "variables": {},
            "lists": {},
            "broadcasts": {},
            "blocks": blocks,
            "comments": {},
            "currentCostume": 0,
            "costumes": [{
                "name": "backdrop1",
                "bitmapResolution": 1,
                "dataFormat": "png",
                "assetId": asset_id,
                "md5ext": asset_id + ".png",
                "rotationCenterX": 0,
                "rotationCenterY": 0,
            }],
            "sounds": [],
            "volume": 100,
            "layerOrder": 0,
            "tempo": 60,
            "videoTransparency": 50,
            "videoState": "on",
            "textToSpeechLanguage": None,
        }
    ],
    "monitors": [],
    "extensions": [],
    "meta": {"semver": "3.0.0", "vm": "0.2.0", "agent": "scratch-torch generator"},
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo-torch.sb3")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("project.json", json.dumps(project, ensure_ascii=False, separators=(",", ":")))
    z.writestr(asset_id + ".png", png)
print("записан", out, "| блоков:", len(blocks))

with zipfile.ZipFile(out) as z:
    pj = json.loads(z.read("project.json"))
print("проверка: blocks keys =", len(pj["targets"][0]["blocks"]))