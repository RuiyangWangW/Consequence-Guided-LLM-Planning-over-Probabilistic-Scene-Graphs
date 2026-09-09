"""Candidate prompts for object extraction, so they can be measured against each other.

Each targets one of the two measured failures. Across the 8B's 43 extraction failures the
model dropped 70 items - 55 of them the objects being moved rather than the furniture -
and returned 2.8 objects where 3.5 were wanted. Separately it named a *room* as an object
16 times, despite one clause telling it not to.

Both look like recall failures rather than reasoning failures, and the baseline prompt has
a plausible cause for each: all eight of its examples answer with two or three objects,
which teaches a short answer; and its room rule is half a sentence inside a paragraph
about something else.

    BASELINE        the short examples alone - the prompt before any of this
    LONG_EXAMPLES   three of the short examples, plus ones the length of real tasks
    RESTRUCTURED    ask for the objects first, classify second, with the room rule promoted
                    to its own line

The winner - every short example kept, long ones added - is now `task_objects.PROMPT`
itself, so it is not repeated here.

Measured, the first two split the models rather than beating each other. Longer examples
took the 8B from 60% to 79% exact by lifting recall 84% -> 94%; the same change left the
4B *worse* than plain normalization (73% vs 79%) because its relation errors rose 15 -> 23.
The 4B was already reading the sentence correctly - recall 98% - so it had nothing to gain
and something to lose. What both winners are then limited by is relations, and both prompts
that hurt relations had dropped examples of them, which is what LONG_PLUS puts back.

Keeping the two separate is what makes the comparison say anything: run together, a gain
could be either.
"""

from task_objects import PROMPT

# The prompt in `task_objects` is the winner, so the variants here are now *ablations* of
# it: what it looks like with a change undone. Deriving them the other way round stopped
# working the moment the winner was promoted - adding the long examples to a prompt that
# already had them produced them twice.
#
# The block of examples runs from the first to the closing rule, both unique strings.
_HEAD, _REST = PROMPT.split("  Task: get the potato", 1)
_EXAMPLES, _TAIL = _REST.split("An object named in a DEPENDENT", 1)
_EXAMPLES = ("  Task: get the potato" + _EXAMPLES).strip("\n").split("\n\n")
_TAIL = "An object named in a DEPENDENT" + _TAIL


def _with(examples):
    return _HEAD + "\n\n".join(examples) + "\n\n" + _TAIL


# The long examples come with the sentence that states outright what they demonstrate, so
# the two travel together: a variant that shows long answers must also say to give them,
# and one that shows neither must say neither.
_LONG = [e for e in _EXAMPLES if "\n        " in e or e.startswith("Some tasks name")]
_SHORT = [e for e in _EXAMPLES if e not in _LONG]

# The eight short examples alone: what the prompt was before any of this. Every one of
# them answers with two or three objects, which is the habit that cost the 8B its recall.
BASELINE = _with(_SHORT)

# Three short examples plus the long ones. Measured worse than keeping all eight: dropping
# five relation examples to make room cost more in INSIDE/ON_TOP errors than the length
# gained in recall (4B 79% -> 73%).
LONG_EXAMPLES = _with([_SHORT[0], _SHORT[6], _SHORT[7]] + _LONG)


RESTRUCTURED = """A household robot has been given a task. List the objects it names.

Work in two steps and show both.

ALL: every physical object the task names, in the order it names them. Include the things
being carried AND the furniture, containers and appliances they are carried between. Copy
the words the task uses. Some tasks name two objects and some name six - list all of them.

Then split ALL into two groups, by whether the task says where the object is NOW:

UNCERTAIN: objects the task names but does not say the current location of.

DEPENDENT: objects whose CURRENT location the task states. Write each as
"<object> INSIDE <container>" or "<object> ON_TOP <container>". Only INSIDE and ON_TOP are
allowed, and both names must be objects from ALL.

Every entry in ALL appears exactly once, in UNCERTAIN or as the <object> of a DEPENDENT
line - never in both. A container named in a relation still belongs in UNCERTAIN, because
we have to go and find it.

DEPENDENT is about where something is NOW, not where the task wants it to end up. "Get the
potato from the fridge" says the potato is in the fridge now, so it is DEPENDENT. "Put the
mug in the dishwasher" describes the goal - the mug is not in the dishwasher yet - so both
are UNCERTAIN and DEPENDENT is empty.

Match the relation to the words used: "inside the cabinet", "from the fridge" and "out of
the fridge" are INSIDE; "on the counter", "on top of the stove" and "off the shelf" are
ON_TOP.

A ROOM IS NOT AN OBJECT. Kitchen, bedroom, bathroom, closet, dining room, living room,
corridor, garage and the like say where the robot goes, never what it handles. Leave them
out of ALL entirely. People are not objects either.

Extract only what the task says. Do NOT add objects it does not name, and do not infer
tools or ingredients - if the task does not mention water, do not list water. Name whole
objects, not parts: "fridge", not "refrigerator_door". When the task names a container as
part of a larger piece of furniture - "the desk's drawer", "the bookcase's cabinet" - the
container is the object, not the furniture around it.

  Task: get the potato from the fridge and cook it on the stove
  ALL: potato, fridge, stove
  UNCERTAIN: fridge, stove
  DEPENDENT: potato INSIDE fridge

  Task: put the mug on top of the stove in the dishwasher
  ALL: mug, stove, dishwasher
  UNCERTAIN: stove, dishwasher
  DEPENDENT: mug ON_TOP stove

  Task: take the towel and the soap from the shelf, put them in the washer, run it,
        then move them to the dryer
  ALL: towel, soap, shelf, washer, dryer
  UNCERTAIN: shelf, washer, dryer
  DEPENDENT: towel ON_TOP shelf, soap ON_TOP shelf

  Task: bring the plate, the bowl and the spoon from the cupboard in the kitchen to the
        table, then wipe the counter with the sponge
  ALL: plate, bowl, spoon, cupboard, table, counter, sponge
  UNCERTAIN: cupboard, table, counter, sponge
  DEPENDENT: plate INSIDE cupboard, bowl INSIDE cupboard, spoon INSIDE cupboard

  Task: put the sketch pad and the ruler in the desk's drawer in the study, then switch
        off the lamp
  ALL: sketch pad, ruler, drawer, lamp
  UNCERTAIN: sketch_pad, ruler, drawer, lamp
  DEPENDENT:

Task: {task}
ALL:"""
