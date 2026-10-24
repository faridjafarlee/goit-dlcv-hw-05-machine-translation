# ДЗ 5 (тема 12): машинний переклад з механізмом уваги, de на en

Здавати: `dz_topic_12_Джафарлі_Фарід.ipynb`. Оцінювання: залік або незалік.

Дані: Helsinki-NLP/europarl, пара de-en, підмножина з 100 тисяч речень для навчання (від 3 до 20 слів), 2 тисячі для валідації, 1 тисяча тестових.

Модель: seq2seq, двонаправлений GRU-енкодер, адитивна увага, GRU-декодер, 28.6 млн параметрів. 8 епох, найкраща 6-та: val loss 2.82 (perplexity 16.8), train loss 2.67. Після 6 епохи val loss росте, це перенавчання, тому залишено модель з 6 епохи.

Інші файли:
- `build_notebook.py` збирає ноутбук;
- `saved_models/seq2seq_de_en.pt` ваги найкращої моделі, `saved_models/history.json` значення втрат по епохах;
- `dump_outputs.py` перевіряє, що в ноутбуці немає помилок (один вивід у stderr: попередження HuggingFace про запит без токена, воно не заважає);
- `old_v1/` попередня, розширена версія.

Перезапуск (близько 55 хвилин на GPU Apple M2 Max; набір даних береться з кешу HuggingFace):

```bash
../.venv/bin/python build_notebook.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 dz_topic_12_Джафарлі_Фарід.ipynb
```
