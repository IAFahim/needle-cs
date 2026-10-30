# Native parity (W29)

rows=24 call mismatches=3 argument mismatches=0 reasoning diffs=3
worst confidence delta=0.0557 ('switch to wedge formation and follow the captain' [<empty>])

| tools | system | query | calls | args | conf delta | decode>=head | reasoning |
|---|---|---|---|---|---|---|---|
| tools.json | <empty> | what's it like in Lagos right now? | same | same | 0.0052 | True | same |
| tools.json | <empty> | set a timer for 12 minutes and play some M | same | same | 0.0026 | True | same |
| tools.json | date: 2026-10-26 Mon 09:41 | what's it like in Lagos right now? | same | same | 0.0107 | True | differ |
| tools.json | date: 2026-10-26 Mon 09:41 | set a timer for 12 minutes and play some M | same | same | 0.0032 | True | same |
| game_tools.json | <empty> | move to 120, 45 | same | same | 0.0037 | True | same |
| game_tools.json | <empty> | attack the orc chieftain | same | same | 0.0082 | True | same |
| game_tools.json | <empty> | give Lyra 3 potions | N:give_item M:say | DIFFER | 0.0049 | True | differ |
| game_tools.json | <empty> | switch to wedge formation and follow the c | same | same | 0.0557 | True | same |
| game_tools.json | <empty> | say hello traveler, the bridge is out | same | same | 0.0146 | True | same |
| game_tools.json | <empty> | stop everything | same | same | 0.0031 | True | same |
| game_tools.json | <empty> | give 50 gold to Borin then move to 10, 20 | same | same | 0.0228 | True | same |
| game_tools.json | <empty> | follow Mira | same | same | 0.0032 | True | same |
| game_tools.json | <empty> | attack the watchtower near the river | N:attack M: | DIFFER | 0.0526 | True | differ |
| game_tools.json | <empty> | form a circle | same | same | 0.0194 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | move to 120, 45 | same | same | 0.0006 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | attack the orc chieftain | same | same | 0.0352 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | give Lyra 3 potions | N:give_item M:say | DIFFER | 0.0473 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | switch to wedge formation and follow the c | same | same | 0.0401 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | say hello traveler, the bridge is out | same | same | 0.0040 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | stop everything | same | same | 0.0107 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | give 50 gold to Borin then move to 10, 20 | same | same | 0.0043 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | follow Mira | same | same | 0.0003 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | attack the watchtower near the river | same | same | 0.0026 | True | same |
| game_tools.json | date: 2026-10-26 Mon 09:41 | form a circle | same | same | 0.0161 | True | same |
