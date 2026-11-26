# Designer agent

Plain-language orders for a squad, wired up in the Inspector with no code.

1. Copy `needle3.cact` into `Assets/StreamingAssets` (run `tools/fetch.sh` in the needle-cs repo, or
   download it from the model page).
2. Create an empty GameObject and add **Needle Agent** (Add Component > Needle > Needle Agent).
3. Drag **Squad Agent** from this sample onto its **Agent** field.
4. Click **Add a binding for every tool**. Each tool gets a **Called** event, and each parameter gets
   an event of the right type: `move_to`'s `x` and `y` arrive as numbers, `give_item`'s `count` as a
   whole number.
5. Wire the events to your scene, the way you wire a Button's On Click.
6. Add **Order Panel** to the same GameObject, press Play and type an order. The Needle Agent
   inspector also has a **Try it** box in Play mode.

The assets:

- **Needle Model** names the model file in StreamingAssets. On Android it is copied out of the APK
  once, on first use.
- **Squad Tools** lists the tools and their parameters. Its inspector warns about names that are not
  snake_case or tools without a description, and shows the JSON the model reads.
- **Squad Agent** joins the two and sets the character: what it always knows (Persona), how sure it
  must be before it acts (Refuse Below, 0.3), what it says when unsure, and how many threads it uses.

The model loads and answers on a thread of its own; the events run on the main thread in `Update`, a
few frames after the order.
