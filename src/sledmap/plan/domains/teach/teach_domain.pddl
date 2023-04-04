; If using VSCode, strongly recommend the PDDL plugin for syntax highlighting + readability + other useful tools.

(define (domain teach)
	(:requirements :adl :action-costs)
	(:types
	    AlarmClock AluminumFoil Apple AppleSliced ArmChair BaseballBat BasketBall Bathtub BathtubBasin Bed Blinds Book Boots Bottle Bowl Box Bread BreadSliced ButterKnife CD Cabinet Candle CellPhone Chair Cloth CoffeeMachine CoffeeTable CounterTop CreditCard Cup Desk DeskLamp Desktop DiningTable DishSponge DogBed Drawer Dresser Dumbbell Egg EggCracked Faucet FloorLamp Footstool Fork Fridge GarbageBag GarbageCan HandTowel HandTowelHolder HousePlant Kettle KeyChain Knife Ladle Laptop LaundryHamper Lettuce LettuceSliced LightSwitch Microwave Mirror Mug Newspaper Ottoman Pan PaperTowelRoll Pen Pencil PepperShaker Pillow Plate Plunger Pot Potato PotatoSliced RemoteControl RoomDecor Safe SaltShaker ScrubBrush Shelf ShelvingUnit ShowerCurtain ShowerDoor ShowerGlass ShowerHead SideTable Sink SinkBasin SoapBar SoapBottle Sofa Spatula Spoon SprayBottle Statue Stool StoveBurner StoveKnob TVStand TableTopDecor TeddyBear Television TennisRacket TissueBox Toaster Toilet ToiletPaper ToiletPaperHanger Tomato TomatoSliced Towel TowelHolder VacuumCleaner Vase Watch WateringCan Window WineBottle - InteractiveObject
        ; Affordances
		AlarmClock AluminumFoil Apple AppleSliced BaseballBat BasketBall Book Boots Bottle Bowl Box Bread BreadSliced ButterKnife Candle CD CellPhone Cloth CreditCard Cup DishSponge Dumbbell Egg EggCracked Fork HandTowel Kettle KeyChain Knife Ladle Laptop Lettuce LettuceSliced Mug Newspaper Pan PaperTowelRoll Pen Pencil PepperShaker Pillow Plate Plunger Pot Potato PotatoSliced RemoteControl SaltShaker ScrubBrush SoapBar SoapBottle Spatula Spoon SprayBottle Statue TableTopDecor TeddyBear TennisRacket TissueBox ToiletPaper Tomato TomatoSliced Towel Vase Watch WateringCan WineBottle - Pickupable
		Apple Bread Egg Lettuce Potato Tomato - Sliceable
		ArmChair Bathtub BathtubBasin Bed Bowl Box Cabinet Chair CoffeeMachine CoffeeTable CounterTop Cup Desk DiningTable Drawer Dresser Fridge GarbageCan HandTowelHolder LaundryHamper Microwave Mug Ottoman Pan Plate Pot Safe Shelf SideTable Sink SinkBasin Sofa Stool StoveBurner Toaster Toilet ToiletPaperHanger TowelHolder TVStand - Receptacle
		ArmChair Chair CoffeeMachine CoffeeTable Desk DeskLamp Desktop DiningTable DogBed Dresser FloorLamp Footstool GarbageBag GarbageCan HousePlant LaundryHamper Microwave Ottoman RoomDecor Safe ShelvingUnit SideTable Sofa Stool Television Toaster TVStand VacuumCleaner - Moveable
		Bed Bowl Cloth Cup Mirror Mug Pan Plate Pot - Dirtyable
		Blinds Book Box Cabinet Drawer Fridge Kettle Laptop Microwave Safe ShowerCurtain ShowerDoor Toilet - Openable
		Bottle Bowl Cup HousePlant Kettle Mug Pot WateringCan WineBottle - Fillable
		Bottle Bowl CellPhone Cup Egg Laptop Mirror Mug Plate ShowerDoor ShowerGlass Statue Television Vase Window WineBottle - Breakable
		Candle CellPhone CoffeeMachine DeskLamp Faucet FloorLamp Laptop LightSwitch Microwave ShowerHead StoveBurner StoveKnob Television Toaster - Toggleable
		PaperTowelRoll SoapBottle TissueBox ToiletPaper - UsedUp
	    BreadSliced EggCracked Potato PotatoSliced - Cookable
	    BreadSliced EggCracked Potato PotatoSliced - Boilable
        BreadSliced EggCracked Potato PotatoSliced - Microwavable
        BreadSliced EggCracked Potato PotatoSliced - Stovecookable
	    BreadSliced - Toastable
	    Chairs Furniture Tables WaterBottomReceptacles OpenableReceptacles - Receptacle
	    Dishware Drinkware Food Fruit HygieneProducts KitchenUtensils Knives Silverware SmallHandheldObjects Soap Tableware - Pickupable
        WaterTaps ManipulableKitchenUtils - Toggleable
	    WaterContainers - Fillable
	    ; Semantic classes
	    Appliances BoilContainers Chairs CleaningProducts Computers Condiments Cookware Dishware Drinkware Electronics Food FoodCookers Fruit Furniture Glass Ground HygieneProducts KitchenUtensils Knives Lamps Lights ManipulableKitchenUtils MediaEntertainment OccupiableReceptacles OpenableReceptacles PickableReceptacles RoomDecor Silverware SmallHandheldObjects Soap SportsEquipment StandingReceptacles StoveTopCookers Tables Tableware Vegetables WallDecor WaterBottomReceptacles WaterContainers WaterSources WaterTaps WritingUtensils - SemanticClass
        CoffeeMachine Fridge Microwave StoveBurner StoveKnob Toaster VacuumCleaner  - Appliances
        ArmChair Chair Sofa Stool  - Chairs
        Cloth DishSponge GarbageBag PaperTowelRoll Plunger ScrubBrush SoapBar SoapBottle SprayBottle TissueBox ToiletPaper VacuumCleaner  - CleaningProducts
        Desktop Laptop  - Computers
        PepperShaker SaltShaker  - Condiments
        Kettle Knife Ladle Pan Pot Spatula  - Cookware
        Bowl Plate  - Dishware
        Bottle Cup Mug  - Drinkware
        AlarmClock CellPhone Desktop Laptop RemoteControl Television  - Electronics
        Apple AppleSliced Tomato TomatoSliced  - Fruit
        ArmChair Bed Chair CoffeeTable Desk DiningTable Dresser Footstool Ottoman Shelf SideTable Sofa Stool TVStand  - Furniture
        HandTowel SoapBar SoapBottle TissueBox Towel  - HygieneProducts
        ButterKnife Fork Knife Ladle Spatula Spoon  - KitchenUtensils
        ButterKnife Knife  - Knives
        DeskLamp FloorLamp  - Lamps
        DeskLamp FloorLamp LightSwitch  - Lights
        Book CD CellPhone Desktop Laptop Newspaper RemoteControl Television  - MediaEntertainment
        AlarmClock Candle HousePlant RoomDecor Statue TableTopDecor TeddyBear Vase WineBottle  - RoomDecorations
        ButterKnife Fork Spoon  - Silverware
        Book CD CellPhone CreditCard KeyChain Pen Pencil RemoteControl Watch  - SmallHandheldObjects
        SoapBar SoapBottle  - Soap
        BaseballBat BasketBall Dumbbell TennisRacket  - SportsEquipment
        CoffeeTable Desk DiningTable Shelf SideTable  - Tables
        Bottle Bowl Cup Mug PepperShaker Plate SaltShaker WineBottle  - Tableware
        Lettuce LettuceSliced Potato PotatoSliced  - Vegetables
        Blinds Curtains Painting Poster ShowerCurtain Window  - WallDecor
        Faucet ShowerHead Toilet  - WaterSources
        Pen Pencil  - WritingUtensils
        CoffeeMachine Microwave StoveKnob Toaster  - ManipulableKitchenUtils
        Bowl Pan Pot Microwave StoveBurner  - FoodCookers
        Kettle Pan Pot  - StoveTopCookers
        Bowl Pot  - BoilContainers
        Faucet ShowerHead  - WaterTaps
        Bathtub BathtubBasin Sink SinkBasin  - WaterBottomReceptacles
        Bottle Bowl Cup Kettle Mug Pot WateringCan WineBottle  - WaterContainers
        Box Cabinet Drawer Fridge Microwave Safe Toilet  - OpenableReceptacles
        ArmChair Bathtub BathtubBasin Bed Cabinet Chair CoffeeMachine CoffeeTable CounterTop Desk DiningTable Drawer Dresser Fridge GarbageCan HandTowelHolder LaundryHamper Microwave Ottoman Safe Shelf SideTable Sink SinkBasin Sofa Stool StoveBurner TVStand Toaster Toilet ToiletPaperHanger TowelHolder  - StandingReceptacles
        Bowl Box Cup Mug Pan Plate Pot  - PickableReceptacles
        Bathtub BathtubBasin Microwave Pan Pot Sink SinkBasin StoveBurner  - OccupiableReceptacles
        Carpet Floor  - Ground
        Mirror Window ShowerGlass  - Glass
        ; Adding other custom affordances
        ; feels a bit redundant for the first three but
        ; it's semantically readable
        AppleSliced LettuceSliced BreadSliced PotatoSliced TomatoSliced - SlicedFood        
	    Agent
	    Location
	    InteractiveObject SemanticClass - Location
	    InteractiveObject SemanticClass - Thing
	)
	
	(:constants
        START_LOC - Location
    )

	(:predicates
	    (atLocation ?a - Agent ?y - InteractiveObject)
        (isInteractable ?x - InteractiveObject)
        (isToggled ?x - Toggleable) ; a bit redundant
        (isBroken ?x - Breakable)
        (isFilledWithLiquid ?x - Fillable)
        (isDirty ?x - Dirtyable)
        (isUsedUp ?x - UsedUp)
        (isCooked ?x - Cookable)
        (isSliced ?x - Sliceable)
        (isOpen ?x - Openable)
        (isInsideClosed ?x - InteractiveObject)
        (isPickedUp ?x - Pickupable)
        (simbotIsPickedUp ?x - Pickupable)
        (parentReceptacles ?x - InteractiveObject ?y - Receptacle) ; x is a child of y
        (simbotIsFilledWithCoffee ?x - Fillable)
        (simbotIsFilledWithWater ?x - Fillable)
        (isObserved ?x - InteractiveObject)
        (simbotIsBoiled ?x - Boilable) ;specifically for eggs...
        ; specific type-level predicates
        (isWaterBottomReceptacles ?x - Receptacle)
        (isBoilable ?x - Boilable)
        (isDirtyable ?x - InteractiveObject)
        (isCookable ?x - Cookable)
        (isFillable ?x - Fillable)
        (isMug ?x - Mug)
        (isCoffeeMachine ?x - Receptacle)    
        (isStoveTopCookers ?x - Receptacle)
        (isBreadSliced ?x - BreadSliced)
        (isToaster ?x - Toaster)
        (isWaterTaps ?x - Toggleable)
        (isStoveBurner ?x - Toggleable)
        (isMicrowave ?x - Toggleable)
        (isClosedOpenable ?x - Openable)
        (isInWater ?x - Pickupable)
        (canBePlacedTo ?x - Pickupable ?y - Receptacle)
        (holding ?x - Pickupable)
        ; special mentions of objects
        ; TODO: add these as conditional effects
        (isBowlOfAppleSlices ?x - Bowl)
        (isBowlOfTomatoSlices ?x - Bowl)
        (isBowlOfLettuceSlices ?x - Bowl)
        (isBowlOfCookedPotatoSlices ?x - Bowl)
        (isPlateOfAppleSlices ?x - Plate)
        (isPlateOfTomatoSlices ?x - Plate)
        (isPlateOfLettuceSlices ?x - Plate)
        (isPlateOfCookedPotatoSlices ?x - Plate)
       (isPlateOfToast ?x - Plate)
        (isSalad ?x - Plate)
        (isSandwich ?x - Plate)
       (isToast ?x - BreadSliced)
        (isCoffee ?x - Mug)
    )
	
	(:functions
        (distance ?from - Location ?to - Location)
        (total-cost)
    )

    ; Goto a visible object to make it interactable
    ; (:action Goto
    ; 	:parameters (?a - Agent ?from - Location ?to - Location)
    ; 	:precondition (and
    ; 	    (isObserved ?to)
    ; 	    (atLocation ?a ?from)
    ; 	    (not (= ?from ?to))
    ; 	    (not (isInteractable ?to))
    ; 	    (not (atLocation ?a ?to))
    ; 	    (not (isClosedOpenable ?to))
    ;     )
    ;     :effect (and
    ;         (isInteractable ?to)
    ;         (atLocation ?a ?to)
    ;         (not (atLocation ?a ?from))
    ;         (forall (?z - InteractiveObject)
    ;             (and
    ;                 (when
    ;                     (parentReceptacles ?z ?to)
    ;                     (isInteractable ?z)
    ;                 )
    ;                 (when
    ;                     (and
    ;                         (not (= ?z ?to))
    ;                         (or (not (parentReceptacles ?z ?to)) (not (parentReceptacles ?to ?z)))
    ;                     )
    ;                     (not (isInteractable ?z))
    ;                 )
    ;             )
    ;         )
    ;         (increase (total-cost) (distance ?from ?to))
    ;         ; (increase (total-cost) 15)
    ;     )
    ; )
    ; Goto a visible object to make it interactable
    (:action Goto
    	:parameters (?a - Agent ?from - Location ?to - Location)
    	:precondition (and
    	    (isObserved ?to)
    	    (atLocation ?a ?from)
    	    (not (= ?from ?to))
    	    (not (isInteractable ?to))
    	    (not (atLocation ?a ?to))
    	    ; (isClosedOpenable ?to)
        )
        :effect (and
            (isInteractable ?to)
            (atLocation ?a ?to)
            (not (atLocation ?a ?from))
            (forall (?z - InteractiveObject)
                (when 
                    (not (= ?z ?to)) 
                    (not (isInteractable ?z))
                )
            )
            (increase (total-cost) (distance ?from ?to))
            ; (increase (total-cost) 15)
        )
    )
    ; Search makes an object become isObserved
    (:action Search
        :parameters (?x - Thing)
        :precondition (and (not (isObserved ?x)) (not (isInsideClosed ?x)))
        :effect (and
            (isObserved ?x)
            (forall (?z - InteractiveObject)
                ; (when
                ;     (not (isPickedUp ?z))
                ;     (not (isInteractable ?z))
                ; )
                (not (isInteractable ?z))
            )
            (increase (total-cost) 100)
        )
    )
    ; Pickup requires that the object is interactable and that no other objects 
    ; are currently picked up.
    ; Contextual effects:
    ; Remove the picked up object from all parents
    ; If the object is a parent, then pick up all the children too
    (:action Pickup
        :parameters (?x - Pickupable)
        :precondition (and
            (isInteractable ?x) 
            (forall (?z - Pickupable) 
                (not (isPickedUp ?z))
            )
            (not 
                (exists (?z - Toggleable) 
                    (and
                        (parentReceptacles ?x ?z)
                        (isToggled ?z)
                    )
                )
            )
        )
        :effect (and
            (holding ?x)
            (isPickedUp ?x)
            ; remove from its parents
            (forall (?z - Receptacle) (not (parentReceptacles ?x ?z)))
            ; children should all be picked up if the parent is picked up
            (forall (?z - Pickupable)
                (when
                    (parentReceptacles ?z ?x)
                    (isPickedUp ?z)
                )
            )
            ; remove children from their parentReceptacles
            ; note: this somehow cannot be conbined with the previous forall 
            (forall (?z - Pickupable)
                (when
                    (parentReceptacles ?z ?x)
                    (forall (?r - Receptacle) 
                        (when
                            (not (= ?r ?x))  
                            (not (parentReceptacles ?z ?r))
                        )
                    )
                )
            )
            (increase (total-cost) 15)
            ; picked object cannot be in water
            (not (isInWater ?x))
        )
    )
    ; Place has many contextual interactions, described below.
    (:action Place
        :parameters (?x - Pickupable ?y - Receptacle);surfaces count as receptacles
        :precondition (and 
            (holding ?x) 
            (isInteractable ?y) 
            (not (isClosedOpenable ?y))
            (canBePlacedTo ?x ?y)
        )
        :effect (and
            (forall (?z - Pickupable) 
                (and (not (isPickedUp ?z)) (not (holding ?z)))
            )
            (forall (?z - Pickupable) 
                (when
                    (isPickedUp ?z)
                    (parentReceptacles ?z ?y)
                )
            )
            (when
                ; want to clean objects when placed under running water
                ; we need to define some kind of hybrid object
                (and 
                    (isDirty ?x) 
                    (isDirtyable ?x) 
                    (isWaterBottomReceptacles ?y) 
                    (exists (?f - WaterTaps)
                        (and
                            (parentReceptacles ?f ?y)
                            (isToggled ?f)
                        )
                    )
                ) ; running water
                (not (isDirty ?x))
            )
            (when
                ; want to fill containers when placed under running water
                (and 
                    (isFillable ?x)
                    (not (simbotIsFilledWithCoffee ?x)) 
                    (not (simbotIsFilledWithWater ?x)) 
                    (isWaterBottomReceptacles ?y)
                    (exists (?f - WaterTaps)
                        (and
                            (parentReceptacles ?f ?y)
                            (isToggled ?f)
                        )
                    )
                )
                (simbotIsFilledWithWater ?x)
            )
            (when
                ; if a mug is placed under a toggled coffeemachine, it fills up
                (and (isMug ?x) (isCoffeeMachine ?y) (isToggled ?y))
                (simbotIsFilledWithCoffee ?x)
            )
            (when
                ; placing a cookable item on top of a pot or pan that is on top
                ; of a toggled stove makes it cooked
                (and 
                    (isCookable ?x) 
                    (isStoveTopCookers ?y) 
                    (exists (?z - StoveBurner)
                        (and
                            (parentReceptacles ?y ?z)
                            (isToggled ?z)
                        )
                    )
                )
                (isCooked ?x)
            )
            (when 
                ; placing a boilable item in a pot with water on top of the
                ; stove makes it boiled
                (and 
                    (isBoilable ?x) 
                    (isStoveTopCookers ?y)
                    (simbotIsFilledWithWater ?y)
                    ; for all stoves want to check whether any of them are 
                    ; a parent of this and also want to check whether it 
                    ; is toggled
                    (exists (?z - StoveBurner)
                        (and
                            (parentReceptacles ?y ?z)
                            (isToggled ?z)
                        )
                    )
                )
                (simbotIsBoiled ?x)
            )
            (when
                ; placing a pot that already has both a boilable and water on
                ; to a stove will make the boilable boiled
                (and 
                    (isStoveTopCookers ?x)
                    (exists (?z - Boilable)
                        (parentReceptacles ?z ?x)
                    )
                    (simbotIsFilledWithWater ?x)
                    (and (isToggled ?y) (isStoveBurner ?y))
                )     
                ; (simbotIsBoiled ?z)        
                (forall (?z - Boilable)
                    (when
                        (parentReceptacles ?z ?x)
                        (simbotIsBoiled ?z)
                    )
                )
            )
            (when
                ; placing a breadslice in a toggled toaster cooks it
                (and
                    (isBreadSliced ?x)
                    (isToaster ?y)
                    (isToggled ?y)
                )
                (isCooked ?x)
            )
            (when 
                (simbotIsFilledWithWater ?y)
                (isInWater ?x)
            )
            (increase (total-cost) 15)
        )
    )
    (:action Open
        :parameters (?x - Openable)
        :precondition (and (isInteractable ?x) (not (isOpen ?x)))
        :effect (and 
            (isOpen ?x) 
            (not (isClosedOpenable ?x))
            ; all child objects becomes interactable and have isInsideClosed set to false
            (forall (?z - InteractiveObject)
                (when
                    (parentReceptacles ?z ?x)
                    (and 
                        (isInteractable ?z)
                        (isObserved ?z)
                        (not (isInsideClosed ?z))
                    )
                )
            )
            (increase (total-cost) 15)
        )
    )
    (:action Close
        :parameters (?x - Openable)
        :precondition (and (isInteractable ?x) (isOpen ?x))
        :effect (and 
            (not (isOpen ?x))
            (isClosedOpenable ?x)
            ; all child objects have isInsideClosed set to true
            (forall (?z - InteractiveObject)
                (when
                    (parentReceptacles ?z ?x)
                    (and (isInsideClosed ?z) (not (isInteractable ?z)))
                )
            )
            (increase (total-cost) 15)
        )
    )
    (:action ToggleOn
        :parameters (?x - Toggleable)
        :precondition (and 
            (isInteractable ?x) 
            (not (isToggled ?x))
            (or (not (isMicrowave ?x)) (not (isOpen ?x)))
        )
        :effect (and
            (isToggled ?x)
            ; can't include all the watertap conditional effects into one 
            ; `when` thanks to Fast Downward not playing nicely with it.
            ; So instead, have three different conditional effects dependent 
            ; on isWaterTap
            (when
                ; when a water tap (faucet, showerhead) is toggled,
                (isWaterTaps ?x)
                ; every dirty instance in the basin will be cleaned
                (forall (?z - Dirtyable)
                    (when
                        (and
                            (isDirty ?z)
                            (exists (?r - WaterBottomReceptacles)
                                (and
                                    (parentReceptacles ?z ?r)
                                    (parentReceptacles ?x ?r)
                                )
                            )
                        )
                        (not (isDirty ?z))
                    )
                )
            )
            (when
                ; when a water tap (faucet, showerhead) is toggled,
                (isWaterTaps ?x)
                ; every container in the basin will be filled with water
                (forall (?z - Fillable)
                    (when
                        (and
                            (exists (?r - WaterBottomReceptacles)
                                (and
                                    (parentReceptacles ?z ?r)
                                    (parentReceptacles ?x ?r)
                                )
                            )
                        )
                        (simbotIsFilledWithWater ?z)
                    )
                )
            )
            (when
                ; when a coffeemachine is toggled and has a mug in it, the mug
                ; will be filled with coffee
                (and 
                    (isCoffeeMachine ?x)
                    ; (exists (?z - Mug)
                    ;     (parentReceptacles ?z ?x)
                    ; )
                )
                (forall (?z - Mug)
                    (when
                        (parentReceptacles ?z ?x) 
                        (simbotIsFilledWithCoffee ?z)
                    )
                )
            )
            (when
                (isStoveBurner ?x)
                ; for all objects on the corresponding stovetop and are filled with water,
                ; for all objects inside these objects
                ; boil those objects
                (forall (?z - StoveTopCookers)
                    (when
                        (and
                            (parentReceptacles ?z ?x)
                            (simbotIsFilledWithWater ?z)
                        )
                        (forall (?y - Boilable)
                            (when 
                                (parentReceptacles ?y ?z)
                                (simbotIsBoiled ?y)
                            )
                        )
                    )
                )
            )
            (when
                (isStoveBurner ?x)
                ; for all objects on the corresponding stovetop
                ; for all objects inside these objects
                ; cook those objects
                (forall (?z - StoveTopCookers)
                    (when
                        (and
                            (parentReceptacles ?z ?x)
                        )
                        (forall (?y - Stovecookable)
                            (when 
                                (parentReceptacles ?y ?z)
                                (isCooked ?y)
                            )
                        )
                    )
                )
            )
            (when
                ; toggle on a microwave will cook and/or boil objects on it 
                ; that are amenable to those actions
                (isMicrowave ?x)
                (forall (?z - Microwavable)
                    (when
                        (parentReceptacles ?z ?x)
                        (isCooked ?z)
                    )
                )
            )
            (when
                ; toggle on a microwave will cook and/or boil objects on it 
                ; that are amenable to those actions
                (isMicrowave ?x)
                (forall (?z - Boilable)
                    (when
                        (and
                            (parentReceptacles ?z ?x)
                            (isInWater ?z)
                        )
                        (simbotIsBoiled ?z)
                    )
                )
            )
            (when
                ; toggle on a toaster will cook the breadslice in it
                (and 
                    (isToaster ?x)
                    (exists (?z - BreadSliced)
                        (parentReceptacles ?z ?x)
                    )
                )
                (forall (?z - Toastable)
                    (when
                        (parentReceptacles ?z ?x)
                        (isCooked ?z)
                    )
                )
            )
            (increase (total-cost) 15)
        )
    )
    (:action ToggleOff
        :parameters (?x - Toggleable)
        :precondition (and 
            (isInteractable ?x) 
            (isToggled ?x)
            (or (not (isMicrowave ?x)) (not (isOpen ?x)))
        )
        :effect (and 
            (not (isToggled ?x))
            ; Note: we do not use stoveknobs to control stoveburners since it is hard to obtain mapping between them. 
            ; (when
            ;     (isStoveKnob ?x)
            ;     (forall (?z - StoveBurner)
            ;         (when
            ;             (parentReceptacles ?z ?x)
            ;             (not (isToggled ?z))
            ;         )
            ;     )
            ; )
            (increase (total-cost) 15)
        )
    )
    ; Slice probably needs editing!
    (:action Slice
        :parameters (?x - Sliceable ?y - Knives)
        :precondition (and (holding ?y) (isInteractable ?x) (not (isSliced ?x)) (not (isPickedUp ?x)))
        :effect (and
            (isSliced ?x)
            (not (isObserved ?x))
            ; change all new slices to observed
            ; (forall (?z - SlicedFood)
            ;     (when
            ;         (sliceParent ?z ?x)
            ;         (isObserved ?z)
            ;     )
            ; )
            ; when an object inside a pot/pan on a toggled stove
            ; is sliced, cook the slice
            ; (when
            ;     (exists (?z - StoveBurner)
            ;         (and 
            ;             (isToggled ?z)
            ;             (exists (?y - StoveTopCookers)
            ;                 (and
            ;                     (parentReceptacles ?y ?z)
            ;                     (parentReceptacles ?x ?y)
            ;                 )
            ;             )
            ;         )
            ;     )
            ;     (forall (?z - SlicedFood)
            ;         (when
            ;             (sliceParent ?z ?x)
            ;             (isCooked ?z)
            ;         )
            ;     )
            ; )
            (increase (total-cost) 15)
        )
    )
    ; Note: currently do not support pouring coffee
    (:action PourToFillable
        :parameters (?x - Fillable ?y - Fillable)
        :precondition (and
             (isPickedUp ?x)
             (simbotIsFilledWithWater ?x)
             (isInteractable ?y)
             (not (simbotIsFilledWithWater ?y))
        )
        :effect (and 
            (simbotIsFilledWithWater ?y) 
            (not (simbotIsFilledWithWater ?x)) 
            (increase (total-cost) 15)
        )
    )
    (:action PourToSink
        :parameters (?x - Fillable ?y - WaterBottomReceptacles)
        :precondition (and (isPickedUp ?x) (simbotIsFilledWithWater ?x) (isInteractable ?y))
        :effect (and 
            (not (simbotIsFilledWithWater ?x)) 
            (increase (total-cost) 15)
        )
    )
)
