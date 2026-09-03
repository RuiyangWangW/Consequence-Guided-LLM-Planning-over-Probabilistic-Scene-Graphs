"""The task definitions: ten scenes, ten tasks each, none shorter than ten actions.

Every task is grounded in the survey in `data/scene_survey.json` - it only names furniture
the scene actually has, in the room its ground truth puts it in. Small objects (`spawn`)
are injected, because BEHAVIOR scenes are furniture-only and nothing in them can be picked
up.

Plans come from `task_shapes.py` rather than being written out, so the length is a property
of the shape and cannot drift while someone edits a plan by hand. A first version of this
file averaged 4.9 actions a task; these average twelve, and the shortest is ten.

`extraction` is the ground-truth answer to stage 1: `uncertain` are the objects the task
names without saying where they are, `dependent` are the ones whose *current* location the
task states. Recording it is what lets an evaluation separate an extraction mistake from a
planning mistake.

`build_tasks.py` replays every plan and refuses to write the dataset unless all of them
apply, meet their goal, and are not satisfied before the robot moves.
"""

from task_shapes import (INSIDE, ON_TOP, carry_two_and_switch, fetch_heat_serve,
                         heat_and_serve, laundry_cycle, load_and_run, move_three,
                         stack_then_store, swap_places, two_into_container, unload_two)


def on(obj, target):
    return {"object": obj, "relation": ON_TOP, "target": target}


def inside(obj, target):
    return {"object": obj, "relation": INSIDE, "target": target}


TASKS = {
    # ---------------------------------------------------------------- Beechwood_0_int
    "Beechwood_0_int": [
        {"task": "take the apple pie from the countertop, heat it in the oven, and put it on "
                 "the breakfast table",
         "extraction": {"uncertain": ["oven", "breakfast_table", "countertop"],
                        "dependent": [on("apple_pie", "countertop")]},
         **heat_and_serve("apple_pie", "countertop", "oven", "breakfast_table")},

        {"task": "take the bottle of soup out of the fridge, warm it in the microwave, and leave "
                 "it on the breakfast table",
         "extraction": {"uncertain": ["fridge", "microwave", "breakfast_table"],
                        "dependent": [inside("bottle_of_soup", "fridge")]},
         **fetch_heat_serve("bottle_of_soup", "fridge", "microwave", "breakfast_table")},

        {"task": "wash the t-shirt in the washer, then dry it in the dryer, leaving both "
                 "machines off and shut",
         "extraction": {"uncertain": ["t_shirt", "washer", "clothes_dryer"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "utility_room_0"},
         **laundry_cycle("t_shirt", "bottom_cabinet", "washer", "clothes_dryer")},

        {"task": "put the mug and the water glass away in the top cabinet, closing it each time",
         "extraction": {"uncertain": ["mug", "water_glass", "top_cabinet", "countertop"],
                        "dependent": []},
         **two_into_container("mug", "water_glass", "countertop", "top_cabinet")},

        {"task": "load the plate and the bowl into the dishwasher and run it",
         "extraction": {"uncertain": ["plate", "bowl", "dishwasher", "countertop"],
                        "dependent": []},
         **load_and_run("plate", "bowl", "countertop", "dishwasher")},

        {"task": "bring the notebook, the magazine and the folder to the bookcase",
         "extraction": {"uncertain": ["notebook", "magazine", "folder", "bookcase"],
                        "dependent": []},
         "rooms": {"bookcase": "living_room_1", "coffee_table": "living_room_1"},
         **move_three(["notebook", "magazine", "folder"],
                      ["coffee_table", "coffee_table", "coffee_table"], "bookcase")},

        {"task": "take the bottle of milk and the bottle of apple juice out of the fridge and put them both on the "
                 "breakfast table",
         "extraction": {"uncertain": ["fridge", "breakfast_table"],
                        "dependent": [inside("bottle_of_milk", "fridge"), inside("bottle_of_apple_juice", "fridge")]},
         **unload_two("bottle_of_milk", "bottle_of_apple_juice", "fridge", "breakfast_table")},

        {"task": "carry the coffee cup and the notebook from the kitchen to the desk in the "
                 "office, then switch the floor lamp on and off again",
         "extraction": {"uncertain": ["coffee_cup", "notebook", "desk", "floor_lamp",
                                      "countertop"],
                        "dependent": []},
         **carry_two_and_switch("coffee_cup", "notebook", ["countertop", "countertop"],
                                "desk", "floor_lamp")},

        {"task": "take the fruitcake out of the fridge, put it on the tray, then put the "
                 "tray away in the bottom cabinet and shut it",
         "extraction": {"uncertain": ["fruitcake", "tray", "bottom_cabinet", "countertop",
                                      "breakfast_table"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "kitchen_0"},
         **stack_then_store("fruitcake", "tray", "fridge", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the vase on the coffee table with the alarm clock on the breakfast "
                 "table, using the countertop to set one down while you move the other",
         "extraction": {"uncertain": ["vase", "alarm_clock", "coffee_table", "breakfast_table",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"coffee_table": "living_room_1", "breakfast_table": "kitchen_0"},
         **swap_places("vase", "alarm_clock", "coffee_table", "breakfast_table", "countertop")},
    ],

    # ---------------------------------------------------------------- Beechwood_1_int
    # No kitchen: bedrooms, two child's rooms, a playroom, a television room, bathrooms.
    "Beechwood_1_int": [
        {"task": "put the notebook and the textbook away in the top cabinet, closing it each time",
         "extraction": {"uncertain": ["notebook", "textbook", "top_cabinet", "breakfast_table"],
                        "dependent": []},
         "rooms": {"breakfast_table": "childs_room_1", "top_cabinet": "childs_room_0"},
         **two_into_container("notebook", "textbook", "breakfast_table", "top_cabinet")},

        {"task": "bring the comic book, the jigsaw puzzle and the crayon to the bookcase in the "
                 "playroom",
         "extraction": {"uncertain": ["comic_book", "jigsaw_puzzle", "crayon", "bookcase"],
                        "dependent": []},
         "rooms": {"bookcase": "playroom_0", "breakfast_table": "playroom_0"},
         **move_three(["comic_book", "jigsaw_puzzle", "crayon"],
                      ["breakfast_table", "breakfast_table", "breakfast_table"],
                      "bookcase")},

        {"task": "take the bath towel and the hand bath towel out of the top cabinet and leave them "
                 "on the bathroom countertop",
         "extraction": {"uncertain": ["top_cabinet", "countertop"],
                        "dependent": [inside("bath_towel", "top_cabinet"),
                                      inside("hand_towel", "top_cabinet")]},
         "rooms": {"top_cabinet": "bathroom_0", "countertop": "bathroom_0"},
         **unload_two("bath_towel", "hand_towel", "top_cabinet", "countertop")},

        {"task": "carry the pillow and the blanket from the sofa to the bed, then turn "
                 "on the table lamp",
         "extraction": {"uncertain": ["pillow", "blanket", "sofa", "bed", "table_lamp"],
                        "dependent": []},
         "rooms": {"bed": "childs_room_0", "table_lamp": "playroom_0"},
         **carry_two_and_switch("pillow", "blanket", ["sofa", "sofa"], "bed",
                                "table_lamp")},

        {"task": "put the toy car and the baseball in the bottom cabinet in the child's room and "
                 "shut it each time",
         "extraction": {"uncertain": ["toy_car", "baseball", "bottom_cabinet", "bed"],
                        "dependent": []},
         "rooms": {"bed": "childs_room_0", "bottom_cabinet": "childs_room_0"},
         **two_into_container("toy_car", "baseball", "bed", "bottom_cabinet")},

        {"task": "take the mug out of the top cabinet, put it on the tray, then store the "
                 "tray in the bottom cabinet and close it",
         "extraction": {"uncertain": ["mug", "tray", "bottom_cabinet", "breakfast_table",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"top_cabinet": "childs_room_0", "countertop": "childs_room_1",
                   "bottom_cabinet": "playroom_0"},
         **stack_then_store("mug", "tray", "top_cabinet", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the lampshade on the bookcase with the picture frame on the breakfast "
                 "table, using the armchair to set one down while you move the other",
         "extraction": {"uncertain": ["lampshade", "picture_frame", "bookcase",
                                      "breakfast_table", "armchair"],
                        "dependent": []},
         "rooms": {"bookcase": "playroom_0", "breakfast_table": "playroom_0",
                   "armchair": "playroom_0"},
         **swap_places("lampshade", "picture_frame", "bookcase", "breakfast_table",
                       "armchair")},

        {"task": "take the folder and the envelope out of the bottom cabinet and put "
                 "them on the breakfast table",
         "extraction": {"uncertain": ["bottom_cabinet", "breakfast_table"],
                        "dependent": [inside("folder", "bottom_cabinet"),
                                      inside("envelope", "bottom_cabinet")]},
         "rooms": {"bottom_cabinet": "playroom_0", "breakfast_table": "playroom_0"},
         **unload_two("folder", "envelope", "bottom_cabinet", "breakfast_table")},

        {"task": "bring the t-shirt, the sock and the hat to the bed in the bedroom",
         "extraction": {"uncertain": ["t_shirt", "sock", "hat", "bed", "bottom_cabinet"],
                        "dependent": []},
         "rooms": {"bed": "bedroom_0", "bottom_cabinet": "bedroom_0"},
         **move_three(["t_shirt", "sock", "hat"],
                      ["bottom_cabinet", "bottom_cabinet", "bottom_cabinet"], "bed")},

        {"task": "put the comic book and the notebook in the bookcase's cabinet in the closet, "
                 "closing it each time",
         "extraction": {"uncertain": ["comic_book", "notebook", "bottom_cabinet", "bookcase"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "closet_0", "bookcase": "closet_0"},
         **two_into_container("comic_book", "notebook", "bookcase", "bottom_cabinet")},
    ],

    # ---------------------------------------------------------------- Benevolence_1_int
    # kitchen (fridge, oven, microwave, dishwasher, countertop, top_cabinet), dining_room_0
    # (breakfast_table, countertop), living_room_0 (sofa, bookcase, cedar_chest,
    # floor_lamp, ottoman), corridor_0 (bookcase, console_table). No cabinets but the
    # kitchen's top_cabinet open.
    "Benevolence_1_int": [
        {"task": "take the casserole from the countertop, bake it in the oven, and put "
                 "it on the breakfast table",
         "extraction": {"uncertain": ["oven", "breakfast_table", "countertop"],
                        "dependent": [on("casserole", "countertop")]},
         "rooms": {"countertop": "kitchen_0"},
         **heat_and_serve("casserole", "countertop", "oven", "breakfast_table")},

        {"task": "take the casserole out of the fridge, heat them in the microwave, and "
                 "leave them on the breakfast table",
         "extraction": {"uncertain": ["fridge", "microwave", "breakfast_table"],
                        "dependent": [inside("casserole", "fridge")]},
         **fetch_heat_serve("casserole", "fridge", "microwave", "breakfast_table")},

        {"task": "load the bowl and the plate into the dishwasher and run it",
         "extraction": {"uncertain": ["bowl", "plate", "dishwasher", "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **load_and_run("bowl", "plate", "countertop", "dishwasher")},

        {"task": "put the basil jar and the can of icetea away in the top cabinet, closing it "
                 "each time",
         "extraction": {"uncertain": ["basil_jar", "can_of_icetea", "top_cabinet",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **two_into_container("basil_jar", "can_of_icetea", "countertop", "top_cabinet")},

        {"task": "bring the notebook, the textbook and the notepad to the bookcase in the living "
                 "room",
         "extraction": {"uncertain": ["notebook", "textbook", "notepad", "bookcase",
                                      "console_table"],
                        "dependent": []},
         "rooms": {"bookcase": "living_room_0"},
         **move_three(["notebook", "textbook", "notepad"],
                      ["console_table", "console_table", "console_table"], "bookcase")},

        {"task": "take the butter and the jar of jam out of the fridge and put them both on the "
                 "breakfast table",
         "extraction": {"uncertain": ["fridge", "breakfast_table"],
                        "dependent": [inside("butter", "fridge"), inside("jar_of_jam", "fridge")]},
         **unload_two("butter", "jar_of_jam", "fridge", "breakfast_table")},

        {"task": "carry the vase and the dip candle from the console table to the breakfast "
                 "table, then switch the floor lamp on and off again",
         "extraction": {"uncertain": ["vase", "dip_candle", "console_table",
                                      "breakfast_table", "floor_lamp"],
                        "dependent": []},
         **carry_two_and_switch("vase", "dip_candle", ["console_table", "console_table"],
                                "breakfast_table", "floor_lamp")},

        {"task": "take the cinnamon roll out of the fridge, put it on the tray, then store "
                 "the tray in the top cabinet and shut it",
         "extraction": {"uncertain": ["cinnamon_roll", "tray", "top_cabinet", "breakfast_table",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **stack_then_store("cinnamon_roll", "tray", "fridge", "countertop",
                            "top_cabinet")},

        {"task": "swap the alarm clock on the console table with the bowl on the countertop, "
                 "using the sofa to set one down while you move the other",
         "extraction": {"uncertain": ["alarm_clock", "bowl", "console_table", "countertop",
                                      "sofa"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **swap_places("alarm_clock", "bowl", "console_table", "countertop", "sofa")},

        {"task": "put the tablespoon and the dinner napkin away in the top cabinet, shutting it "
                 "each time",
         "extraction": {"uncertain": ["tablespoon", "dinner_napkin", "top_cabinet",
                                      "breakfast_table"],
                        "dependent": []},
         **two_into_container("tablespoon", "dinner_napkin", "breakfast_table", "top_cabinet")},
    ],

    # ---------------------------------------------------------------- Ihlen_1_int
    "Ihlen_1_int": [
        {"task": "take the pizza from the countertop, bake it in the oven, and put it on the "
                 "breakfast table in the dining room",
         "extraction": {"uncertain": ["oven", "breakfast_table", "countertop"],
                        "dependent": [on("pizza", "countertop")]},
         "rooms": {"countertop": "kitchen_0", "breakfast_table": "dining_room_0"},
         **heat_and_serve("pizza", "countertop", "oven", "breakfast_table")},

        {"task": "take the casserole out of the fridge, heat it in the oven, and leave it on "
                 "the breakfast table in the dining room",
         "extraction": {"uncertain": ["fridge", "oven", "breakfast_table"],
                        "dependent": [inside("casserole", "fridge")]},
         "rooms": {"breakfast_table": "dining_room_0"},
         **fetch_heat_serve("casserole", "fridge", "oven", "breakfast_table")},

        {"task": "load the water glass and the mug into the dishwasher and start it",
         "extraction": {"uncertain": ["water_glass", "mug", "dishwasher", "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **load_and_run("water_glass", "mug", "countertop", "dishwasher")},

        {"task": "put the bag of flour and the sugar sack away in the top cabinet, closing it each time",
         "extraction": {"uncertain": ["bag_of_flour", "sugar_sack", "top_cabinet", "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **two_into_container("bag_of_flour", "sugar_sack", "countertop", "top_cabinet")},

        {"task": "bring the textbook, the notebook and the newspaper to the bookcase in "
                 "the kitchen",
         "extraction": {"uncertain": ["textbook", "notebook", "newspaper", "bookcase",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0", "bookcase": "kitchen_0"},
         **move_three(["textbook", "notebook", "newspaper"],
                      ["countertop", "countertop", "countertop"], "bookcase")},

        # A `cedar_chest` is not in `planner.OPENABLE`, so `unload_two` cannot use it -
        # the shape opens the container. The verifier caught it; the bedroom's bottom
        # cabinet does have a door.
        {"task": "take the sweatshirt and the scarf out of the bottom cabinet and put them "
                 "on the bed",
         "extraction": {"uncertain": ["bottom_cabinet", "bed"],
                        "dependent": [inside("sweatshirt", "bottom_cabinet"),
                                      inside("scarf", "bottom_cabinet")]},
         "rooms": {"bottom_cabinet": "bedroom_0", "bed": "bedroom_0"},
         **unload_two("sweatshirt", "scarf", "bottom_cabinet", "bed")},

        {"task": "carry the mug and the plate from the kitchen to the coffee table, then "
                 "switch on the floor lamp",
         "extraction": {"uncertain": ["mug", "plate", "coffee_table", "floor_lamp",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0"},
         **carry_two_and_switch("mug", "plate", ["countertop", "countertop"],
                                "coffee_table", "floor_lamp")},

        {"task": "take the butter out of the fridge, put it on the tray, then store the tray "
                 "in the bottom cabinet and close it",
         "extraction": {"uncertain": ["butter", "tray", "bottom_cabinet",
                                      "breakfast_table", "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0", "bottom_cabinet": "kitchen_0",
                   "breakfast_table": "dining_room_0"},
         **stack_then_store("butter", "tray", "fridge", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the pillow on the sofa with the blanket on the armchair, using the "
                 "coffee table to set one down while you move the other",
         "extraction": {"uncertain": ["pillow", "blanket", "sofa", "armchair",
                                      "coffee_table"],
                        "dependent": []},
         **swap_places("pillow", "blanket", "sofa", "armchair", "coffee_table")},

        {"task": "put the tablecloth and the place mat away in the bottom cabinet, "
                 "shutting it each time",
         "extraction": {"uncertain": ["tablecloth", "place_mat", "bottom_cabinet",
                                      "breakfast_table"],
                        "dependent": []},
         "rooms": {"breakfast_table": "dining_room_0", "bottom_cabinet": "kitchen_0"},
         **two_into_container("tablecloth", "place_mat", "breakfast_table",
                              "bottom_cabinet")},
    ],

    # ---------------------------------------------------------------- Merom_1_int
    # The kitchen has a stove rather than an oven and no countertop: fridge, dishwasher,
    # stove, cabinets, trash can. bedroom_0 has a hamper (no door, so PLACE_INSIDE needs
    # no OPEN); living_room_0 a sofa, coffee table and floor lamp.
    "Merom_1_int": [
        {"task": "load the saucepan and the frying pan into the dishwasher and run it",
         "extraction": {"uncertain": ["saucepan", "frying_pan", "dishwasher", "stove"],
                        "dependent": []},
         **load_and_run("saucepan", "frying_pan", "stove", "dishwasher")},

        {"task": "take the egg and the bottle of milk out of the fridge and put them both on the "
                 "breakfast table",
         "extraction": {"uncertain": ["fridge", "breakfast_table"],
                        "dependent": [inside("egg", "fridge"), inside("bottle_of_milk", "fridge")]},
         **unload_two("egg", "bottle_of_milk", "fridge", "breakfast_table")},

        {"task": "put the box of cereal and the bag of rice away in the top cabinet in the kitchen, "
                 "closing it each time",
         "extraction": {"uncertain": ["box_of_cereal", "bag_of_rice", "top_cabinet", "bottom_cabinet"],
                        "dependent": []},
         "rooms": {"top_cabinet": "kitchen_0", "bottom_cabinet": "kitchen_0"},
         **two_into_container("box_of_cereal", "bag_of_rice", "bottom_cabinet", "top_cabinet")},

        {"task": "bring the sock, the t-shirt and the bath towel to the hamper in the bedroom",
         "extraction": {"uncertain": ["sock", "t_shirt", "bath_towel", "hamper", "bed"],
                        "dependent": []},
         "rooms": {"bed": "bedroom_0"},
         **move_three(["sock", "t_shirt", "bath_towel"], ["bed", "bed", "bed"], "hamper")},

        {"task": "carry the gaming controller and the magazine from the coffee table to the child's "
                 "bed, then switch the table lamp on and off again",
         "extraction": {"uncertain": ["gaming_controller", "magazine", "coffee_table", "bed",
                                      "table_lamp"],
                        "dependent": []},
         "rooms": {"bed": "childs_room_0", "table_lamp": "childs_room_0"},
         **carry_two_and_switch("gaming_controller", "magazine", ["coffee_table", "coffee_table"],
                                "bed", "table_lamp")},

        {"task": "put the bar soap and the shampoo water bottle away in the bathroom top cabinet, "
                 "shutting it each time",
         "extraction": {"uncertain": ["bar_soap", "shampoo_bottle", "top_cabinet",
                                      "furniture_sink"],
                        "dependent": []},
         "rooms": {"top_cabinet": "bathroom_0", "furniture_sink": "bathroom_0"},
         **two_into_container("bar_soap", "shampoo_bottle", "furniture_sink", "top_cabinet")},

        {"task": "take the plate and the bowls out of the bottom cabinet and put them "
                 "on the breakfast table",
         "extraction": {"uncertain": ["bottom_cabinet", "breakfast_table"],
                        "dependent": [inside("plate", "bottom_cabinet"),
                                      inside("bowl", "bottom_cabinet")]},
         "rooms": {"bottom_cabinet": "kitchen_0"},
         **unload_two("plate", "bowl", "bottom_cabinet", "breakfast_table")},

        {"task": "take the mug out of the top cabinet, put it on the tray, then store the "
                 "tray in the kitchen bottom cabinet and close it",
         "extraction": {"uncertain": ["mug", "tray", "bottom_cabinet", "breakfast_table",
                                      "coffee_table"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "kitchen_0"},
         **stack_then_store("mug", "tray", "top_cabinet", "breakfast_table",
                            "bottom_cabinet")},

        {"task": "swap the teddy bear on the ottoman with the pillow on the sofa, using the "
                 "coffee table to set one down while you move the other",
         "extraction": {"uncertain": ["teddy_bear", "pillow", "ottoman", "sofa",
                                      "coffee_table"],
                        "dependent": []},
         **swap_places("teddy_bear", "pillow", "ottoman", "sofa", "coffee_table")},

        {"task": "bring the wrapping paper, the carton and the can to the trash can in the "
                 "kitchen",
         "extraction": {"uncertain": ["wrapping_paper", "carton", "can", "public_trash_can",
                                      "bottom_cabinet"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "kitchen_0", "public_trash_can": "kitchen_0"},
         **move_three(["wrapping_paper", "carton", "can"],
                      ["bottom_cabinet", "bottom_cabinet", "bottom_cabinet"],
                      "public_trash_can")},
    ],

    # ---------------------------------------------------------------- Pomaria_0_int
    # No kitchen and no appliances beyond the television: living_room_0 (sofa, armchair,
    # coffee_table, standing_tv, bookcase), private_office_0 (countertop, breakfast_table,
    # bottom_cabinet, bookcase), two bedrooms, corridor_0 (bookcase).
    "Pomaria_0_int": [
        {"task": "put the stapler and the duct tape away in the bottom cabinet, closing it "
                 "each time",
         "extraction": {"uncertain": ["stapler", "duct_tape", "bottom_cabinet", "countertop"],
                        "dependent": []},
         **two_into_container("stapler", "duct_tape", "countertop", "bottom_cabinet")},

        {"task": "take the charger and the duct tape out of the bottom cabinet and "
                 "leave them on the countertop",
         "extraction": {"uncertain": ["bottom_cabinet", "countertop"],
                        "dependent": [inside("charger", "bottom_cabinet"),
                                      inside("duct_tape", "bottom_cabinet")]},
         **unload_two("charger", "duct_tape", "bottom_cabinet", "countertop")},

        {"task": "bring the textbook, the notepad and the notebook to the bookcase in the "
                 "corridor",
         "extraction": {"uncertain": ["textbook", "notepad", "notebook", "bookcase",
                                      "coffee_table"],
                        "dependent": []},
         "rooms": {"bookcase": "corridor_0"},
         **move_three(["textbook", "notepad", "notebook"],
                      ["coffee_table", "coffee_table", "coffee_table"], "bookcase")},

        {"task": "carry the mug and the plate from the coffee table to the office "
                 "countertop, then turn the standing tv on and off again",
         "extraction": {"uncertain": ["mug", "plate", "coffee_table", "countertop",
                                      "standing_tv"],
                        "dependent": []},
         **carry_two_and_switch("mug", "plate", ["coffee_table", "coffee_table"],
                                "countertop", "standing_tv")},

        {"task": "put the cardstock and the folder away in the office bottom cabinet, "
                 "closing it each time",
         "extraction": {"uncertain": ["cardstock", "folder", "bottom_cabinet",
                                      "breakfast_table", "countertop"],
                        "dependent": []},
         "rooms": {"breakfast_table": "bedroom_1"},
         **two_into_container("cardstock", "folder", "breakfast_table",
                              "bottom_cabinet")},

        {"task": "swap the pillow on the sofa with the blanket on the armchair, using the "
                 "coffee table to set one down while you move the other",
         "extraction": {"uncertain": ["pillow", "blanket", "sofa", "armchair",
                                      "coffee_table"],
                        "dependent": []},
         **swap_places("pillow", "blanket", "sofa", "armchair", "coffee_table")},

        {"task": "bring the pillow, the blanket and the bath towel to the bed in the bedroom",
         "extraction": {"uncertain": ["pillow", "blanket", "bath_towel", "bed", "sofa"],
                        "dependent": []},
         "rooms": {"bed": "bedroom_0"},
         **move_three(["pillow", "blanket", "bath_towel"], ["sofa", "sofa", "sofa"], "bed")},

        {"task": "put the pen and the cardstock away in the office bottom cabinet, shutting "
                 "it each time",
         "extraction": {"uncertain": ["pen", "cardstock", "bottom_cabinet",
                                      "breakfast_table"],
                        "dependent": []},
         "rooms": {"breakfast_table": "private_office_0"},
         **two_into_container("pen", "cardstock", "breakfast_table", "bottom_cabinet")},

        # The scene has its own guitar, so spawning one would duplicate the node.
        {"task": "carry the cardstock and the notepad from the bed to the office countertop, "
                 "then switch the standing tv on and off again",
         "extraction": {"uncertain": ["cardstock", "notepad", "bed", "countertop",
                                      "standing_tv"],
                        "dependent": []},
         "rooms": {"bed": "bedroom_1"},
         **carry_two_and_switch("cardstock", "notepad", ["bed", "bed"], "countertop",
                                "standing_tv")},

        {"task": "take the folder and the envelope out of the bottom cabinet and put "
                 "them on the breakfast table",
         "extraction": {"uncertain": ["bottom_cabinet", "breakfast_table"],
                        "dependent": [inside("folder", "bottom_cabinet"),
                                      inside("envelope", "bottom_cabinet")]},
         "rooms": {"breakfast_table": "private_office_0"},
         **unload_two("folder", "envelope", "bottom_cabinet", "breakfast_table")},
    ],

    # ---------------------------------------------------------------- Pomaria_1_int
    # A full kitchen, a pantry with a second fridge, and a utility room with washer and
    # dryer - the only scene besides Wainscott_1_int that can run a laundry cycle.
    "Pomaria_1_int": [
        {"task": "take the cinnamon roll from the countertop, bake it in the oven, and put it on "
                 "the breakfast table",
         "extraction": {"uncertain": ["oven", "breakfast_table", "countertop"],
                        "dependent": [on("cinnamon_roll", "countertop")]},
         "rooms": {"countertop": "kitchen_0"},
         **heat_and_serve("cinnamon_roll", "countertop", "oven", "breakfast_table")},

        {"task": "take the casserole out of the fridge, heat them in the microwave, and "
                 "leave them on the breakfast table",
         "extraction": {"uncertain": ["fridge", "microwave", "breakfast_table"],
                        "dependent": [inside("casserole", "fridge")]},
         "rooms": {"fridge": "kitchen_0"},
         **fetch_heat_serve("casserole", "fridge", "microwave", "breakfast_table")},

        {"task": "wash the bath towels in the washer, then dry them in the dryer, leaving both "
                 "machines off and shut",
         "extraction": {"uncertain": ["bath_towel", "washer", "clothes_dryer"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "utility_room_0"},
         **laundry_cycle("bath_towel", "bottom_cabinet", "washer", "clothes_dryer")},

        {"task": "load the frying pan and the crock pot into the dishwasher and run it",
         "extraction": {"uncertain": ["frying_pan", "crock_pot", "dishwasher", "burner"],
                        "dependent": []},
         **load_and_run("frying_pan", "crock_pot", "burner", "dishwasher")},

        {"task": "put the bag of flour and the sugar sack away in the pantry top cabinet, closing it "
                 "each time",
         "extraction": {"uncertain": ["bag_of_flour", "sugar_sack", "top_cabinet", "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0", "top_cabinet": "pantry_room_0"},
         **two_into_container("bag_of_flour", "sugar_sack", "countertop", "top_cabinet")},

        {"task": "take the jar of jam and the swiss cheese out of the pantry fridge and put them on "
                 "the breakfast table",
         "extraction": {"uncertain": ["fridge", "breakfast_table"],
                        "dependent": [inside("jar_of_jam", "fridge"), inside("swiss_cheese", "fridge")]},
         "rooms": {"fridge": "pantry_room_0"},
         **unload_two("jar_of_jam", "swiss_cheese", "fridge", "breakfast_table")},

        {"task": "bring the newspaper, the notebook and the magazine to the bookcase in the "
                 "living room",
         "extraction": {"uncertain": ["newspaper", "notebook", "magazine", "bookcase",
                                      "coffee_table"],
                        "dependent": []},
         "rooms": {"bookcase": "living_room_0", "coffee_table": "living_room_0"},
         **move_three(["newspaper", "notebook", "magazine"],
                      ["coffee_table", "coffee_table", "coffee_table"], "bookcase")},

        {"task": "carry the mug and the bowl from the kitchen countertop to the corridor "
                 "coffee table, then run the kitchen furniture sink and turn it off",
         "extraction": {"uncertain": ["mug", "bowl", "countertop", "coffee_table",
                                      "furniture_sink"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0", "coffee_table": "corridor_0",
                   "furniture_sink": "kitchen_0"},
         **carry_two_and_switch("mug", "bowl", ["countertop", "countertop"],
                                "coffee_table", "furniture_sink")},

        {"task": "take the fruitcake out of the fridge, put it on the tray, then store the "
                 "tray in the kitchen bottom cabinet and shut it",
         "extraction": {"uncertain": ["fruitcake", "tray", "bottom_cabinet",
                                      "breakfast_table", "countertop"],
                        "dependent": []},
         "rooms": {"countertop": "kitchen_0", "bottom_cabinet": "kitchen_0"},
         **stack_then_store("fruitcake", "tray", "fridge", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the detergent bottle in the utility room bottom cabinet with the bar "
                 "soap on the bathroom furniture sink, using the kitchen countertop to set "
                 "one down",
         "extraction": {"uncertain": ["detergent_bottle", "bar_soap", "bottom_cabinet",
                                      "furniture_sink", "countertop"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "utility_room_0", "furniture_sink": "bathroom_0",
                   "countertop": "kitchen_0"},
         **swap_places("detergent_bottle", "bar_soap", "bottom_cabinet", "furniture_sink",
                       "countertop")},
    ],

    # ---------------------------------------------------------------- Rs_int
    # A small apartment. The kitchen holds the appliances but no countertop - the
    # countertop, breakfast table, coffee table, sofa, TV and the scene's own laptop are
    # all in living_room_0.
    "Rs_int": [
        {"task": "take the sliced roast beef from the countertop, cook it in the oven, and put it on "
                 "the breakfast table",
         "extraction": {"uncertain": ["oven", "breakfast_table", "countertop"],
                        "dependent": [on("sliced_roast_beef", "countertop")]},
         **heat_and_serve("sliced_roast_beef", "countertop", "oven", "breakfast_table")},

        {"task": "take the club sandwich out of the fridge, heat it in the microwave, and "
                 "leave it on the countertop",
         "extraction": {"uncertain": ["fridge", "microwave", "countertop"],
                        "dependent": [inside("club_sandwich", "fridge")]},
         **fetch_heat_serve("club_sandwich", "fridge", "microwave", "countertop")},

        {"task": "load the bowl and the plate into the dishwasher and start it",
         "extraction": {"uncertain": ["bowl", "plate", "dishwasher", "countertop"],
                        "dependent": []},
         **load_and_run("bowl", "plate", "countertop", "dishwasher")},

        {"task": "put the box of cereal and the bag of rice away in the kitchen top cabinet, closing "
                 "it each time",
         "extraction": {"uncertain": ["box_of_cereal", "bag_of_rice", "top_cabinet", "countertop"],
                        "dependent": []},
         **two_into_container("box_of_cereal", "bag_of_rice", "countertop", "top_cabinet")},

        {"task": "bring the banana, the wrapping paper and the carton to the kitchen trash "
                 "can",
         "extraction": {"uncertain": ["banana", "wrapping_paper", "carton",
                                      "public_trash_can", "countertop"],
                        "dependent": []},
         **move_three(["banana", "wrapping_paper", "carton"],
                      ["countertop", "countertop", "countertop"], "public_trash_can")},

        {"task": "take the yogurt carton and the bottle of milk out of the fridge and put them both on "
                 "the breakfast table",
         "extraction": {"uncertain": ["fridge", "breakfast_table"],
                        "dependent": [inside("yogurt_carton", "fridge"), inside("bottle_of_milk", "fridge")]},
         **unload_two("yogurt_carton", "bottle_of_milk", "fridge", "breakfast_table")},

        # Rs_int has a `laptop` of its own, so spawning one would put two nodes of that
        # name in the graph. These are objects the scene does not already hold.
        {"task": "carry the notebook and the notepad to the breakfast table, then switch "
                 "the floor lamp on and off again",
         "extraction": {"uncertain": ["notebook", "notepad", "breakfast_table",
                                      "floor_lamp", "countertop"],
                        "dependent": []},
         **carry_two_and_switch("notebook", "notepad", ["countertop", "countertop"],
                                "breakfast_table", "floor_lamp")},

        {"task": "put the gym shoe away in the entrance cabinet along with the umbrella, "
                 "closing it each time",
         "extraction": {"uncertain": ["gym_shoe", "umbrella", "bottom_cabinet", "ottoman"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "entryway_0"},
         **two_into_container("gym_shoe", "umbrella", "ottoman", "bottom_cabinet")},

        {"task": "take the mug out of the top cabinet, put it on the tray, then store the "
                 "tray in the kitchen bottom cabinet and close it",
         "extraction": {"uncertain": ["mug", "tray", "bottom_cabinet", "coffee_table",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "kitchen_0"},
         **stack_then_store("mug", "tray", "top_cabinet", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the pillow on the bed with the blanket on the sofa, using the ottoman "
                 "to set one down while you move the other",
         "extraction": {"uncertain": ["pillow", "blanket", "bed", "sofa", "ottoman"],
                        "dependent": []},
         **swap_places("pillow", "blanket", "bed", "sofa", "ottoman")},
    ],

    # ---------------------------------------------------------------- Wainscott_0_int
    # This scene's standable floor comes in two regions the robot cannot drive between, so
    # every task here stays inside the kitchen / dining_room_0 / living_room_0 /
    # living_room_1 region - the richer of the two. See `data/scene_survey.json`.
    "Wainscott_0_int": [
        {"task": "take the casserole from the countertop, cook it in the oven, and put it on "
                 "the breakfast table in the dining room",
         "extraction": {"uncertain": ["oven", "breakfast_table", "countertop"],
                        "dependent": [on("casserole", "countertop")]},
         "rooms": {"breakfast_table": "dining_room_0"},
         **heat_and_serve("casserole", "countertop", "oven", "breakfast_table")},

        {"task": "take the swiss cheese out of the fridge, warm it in the microwave, and "
                 "leave it on the breakfast table in the dining room",
         "extraction": {"uncertain": ["fridge", "microwave", "breakfast_table"],
                        "dependent": [inside("swiss_cheese", "fridge")]},
         "rooms": {"breakfast_table": "dining_room_0"},
         **fetch_heat_serve("swiss_cheese", "fridge", "microwave", "breakfast_table")},

        {"task": "load the crock pot and the frying pan into the dishwasher and run it",
         "extraction": {"uncertain": ["crock_pot", "frying_pan", "dishwasher", "stove"],
                        "dependent": []},
         **load_and_run("crock_pot", "frying_pan", "stove", "dishwasher")},

        {"task": "put the can of icetea and the can of coffee away in the top cabinet, closing it "
                 "each time",
         "extraction": {"uncertain": ["can_of_icetea", "can_of_coffee", "top_cabinet",
                                      "countertop"],
                        "dependent": []},
         **two_into_container("can_of_icetea", "can_of_coffee", "countertop", "top_cabinet")},

        {"task": "bring the textbook, the newspaper and the notebook to the bookcase in "
                 "the kitchen",
         "extraction": {"uncertain": ["textbook", "newspaper", "notebook", "bookcase",
                                      "countertop"],
                        "dependent": []},
         **move_three(["textbook", "newspaper", "notebook"],
                      ["countertop", "countertop", "countertop"], "bookcase")},

        {"task": "take the bottle of milk and the butter out of the fridge and put them both "
                 "on the breakfast table in the dining room",
         "extraction": {"uncertain": ["fridge", "breakfast_table"],
                        "dependent": [inside("bottle_of_milk", "fridge"), inside("butter", "fridge")]},
         "rooms": {"breakfast_table": "dining_room_0"},
         **unload_two("bottle_of_milk", "butter", "fridge", "breakfast_table")},

        {"task": "carry the mug and the plate from the kitchen to the living room coffee "
                 "table, then run the coffee maker and switch it off",
         "extraction": {"uncertain": ["mug", "plate", "countertop", "coffee_table",
                                      "coffee_maker"],
                        "dependent": []},
         "rooms": {"coffee_table": "living_room_1"},
         **carry_two_and_switch("mug", "plate", ["countertop", "countertop"],
                                "coffee_table", "coffee_maker")},

        {"task": "take the fruitcake out of the fridge, put it on the tray, then store the "
                 "tray in the dining room bottom cabinet and shut it",
         "extraction": {"uncertain": ["fruitcake", "tray", "bottom_cabinet",
                                      "breakfast_table", "countertop"],
                        "dependent": []},
         "rooms": {"breakfast_table": "dining_room_0", "bottom_cabinet": "dining_room_0"},
         **stack_then_store("fruitcake", "tray", "fridge", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the pillow on the sofa with the blanket on the armchair in the living "
                 "room, using the coffee table to set one down while you move the other",
         "extraction": {"uncertain": ["pillow", "blanket", "sofa", "armchair",
                                      "coffee_table"],
                        "dependent": []},
         "rooms": {"sofa": "living_room_1", "armchair": "living_room_1",
                   "coffee_table": "living_room_1"},
         **swap_places("pillow", "blanket", "sofa", "armchair", "coffee_table")},

        {"task": "put the place mat and the dinner napkin away in the dining room cabinet, "
                 "shutting it each time",
         "extraction": {"uncertain": ["place_mat", "dinner_napkin", "bottom_cabinet",
                                      "breakfast_table"],
                        "dependent": []},
         "rooms": {"breakfast_table": "dining_room_0", "bottom_cabinet": "dining_room_0"},
         **two_into_container("place_mat", "dinner_napkin", "breakfast_table",
                              "bottom_cabinet")},
    ],

    # ---------------------------------------------------------------- Wainscott_1_int
    # Upstairs, no kitchen: three bedrooms, a playroom with a pool table, an exercise room
    # and a utility room with washer and dryer.
    "Wainscott_1_int": [
        {"task": "wash the t-shirt in the washer, then dry it in the dryer, leaving both "
                 "machines off and shut",
         "extraction": {"uncertain": ["t_shirt", "washer", "clothes_dryer"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "utility_room_0"},
         **laundry_cycle("t_shirt", "bottom_cabinet", "washer", "clothes_dryer")},

        {"task": "wash the bath towels in the washer, then dry them in the dryer, leaving both "
                 "machines off and shut",
         "extraction": {"uncertain": ["bath_towel", "washer", "clothes_dryer"],
                        "dependent": []},
         "rooms": {"top_cabinet": "utility_room_0"},
         **laundry_cycle("bath_towel", "top_cabinet", "washer", "clothes_dryer")},

        {"task": "put the board game and the jigsaw puzzle away in the playroom top cabinet, "
                 "closing it each time",
         "extraction": {"uncertain": ["board_game", "jigsaw_puzzle", "top_cabinet",
                                      "coffee_table"],
                        "dependent": []},
         "rooms": {"top_cabinet": "playroom_0", "coffee_table": "playroom_0"},
         **two_into_container("board_game", "jigsaw_puzzle", "coffee_table", "top_cabinet")},

        {"task": "bring the baseball, the jigsaw puzzle and the comic book to the pool "
                 "table",
         "extraction": {"uncertain": ["baseball", "jigsaw_puzzle", "comic_book",
                                      "pool_table", "breakfast_table"],
                        "dependent": []},
         "rooms": {"breakfast_table": "playroom_0"},
         **move_three(["baseball", "jigsaw_puzzle", "comic_book"],
                      ["breakfast_table", "breakfast_table", "breakfast_table"],
                      "pool_table")},

        {"task": "take the toothbrush and the tube of toothpaste out of the bathroom cabinet and "
                 "leave them on the countertop",
         "extraction": {"uncertain": ["bottom_cabinet", "countertop"],
                        "dependent": [inside("toothbrush", "bottom_cabinet"),
                                      inside("tube_of_toothpaste", "bottom_cabinet")]},
         "rooms": {"bottom_cabinet": "bathroom_0", "countertop": "bathroom_0"},
         **unload_two("toothbrush", "tube_of_toothpaste", "bottom_cabinet", "countertop")},

        {"task": "carry the water bottle and the bath towel from the bedroom to the treadmill, then "
                 "switch on the table lamp",
         "extraction": {"uncertain": ["water_bottle", "bath_towel", "breakfast_table", "treadmill",
                                      "table_lamp"],
                        "dependent": []},
         "rooms": {"breakfast_table": "bedroom_0", "table_lamp": "bedroom_0"},
         **carry_two_and_switch("water_bottle", "bath_towel", ["breakfast_table", "breakfast_table"],
                                "treadmill", "table_lamp")},

        {"task": "put the pillow and the blanket away in the bedroom cabinet, shutting "
                 "it each time",
         "extraction": {"uncertain": ["pillow", "blanket", "bottom_cabinet", "bed"],
                        "dependent": []},
         "rooms": {"bottom_cabinet": "bedroom_1", "bed": "bedroom_1"},
         **two_into_container("pillow", "blanket", "bed", "bottom_cabinet")},

        {"task": "take the bar soap out of the top cabinet, put it on the tray, then store "
                 "the tray in the bathroom bottom cabinet and close it",
         "extraction": {"uncertain": ["bar_soap", "tray", "bottom_cabinet", "furniture_sink",
                                      "countertop"],
                        "dependent": []},
         "rooms": {"top_cabinet": "utility_room_0", "countertop": "bathroom_0",
                   "bottom_cabinet": "bathroom_0"},
         **stack_then_store("bar_soap", "tray", "top_cabinet", "countertop",
                            "bottom_cabinet")},

        {"task": "swap the notebook on the coffee table with the lampshade on the breakfast "
                 "table, using the pool table to set one down while you move the other",
         "extraction": {"uncertain": ["notebook", "lampshade", "coffee_table",
                                      "breakfast_table", "pool_table"],
                        "dependent": []},
         "rooms": {"coffee_table": "playroom_0", "breakfast_table": "playroom_0"},
         **swap_places("notebook", "lampshade", "coffee_table", "breakfast_table",
                       "pool_table")},

        {"task": "bring the box of tissues, the wrapping paper and the water bottle to the "
                 "bathroom trash can",
         "extraction": {"uncertain": ["box_of_tissues", "wrapping_paper", "water_bottle", "public_trash_can",
                                      "furniture_sink"],
                        "dependent": []},
         "rooms": {"furniture_sink": "bathroom_1", "public_trash_can": "bathroom_1"},
         **move_three(["box_of_tissues", "wrapping_paper", "water_bottle"],
                      ["furniture_sink", "furniture_sink", "furniture_sink"],
                      "public_trash_can")},
    ],
}
