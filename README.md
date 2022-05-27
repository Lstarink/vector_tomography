# vector_tomography

Hoi Erol en Willem, 

Als jullie dit bestand gelezen hebben begrijpen jullie hopelijk genoeg van het model om er mee te kunnen werken!
Eerst een paar stappen om alles te downloaden:

1: Allereerst heb je anaconda navigator nodig. Erol heeft dit waarschijnlijk al, en Willem je kunt het downloaden in brightspace via de software portal.
2: Daarnaast heb je PyCharm community nodig. Dit kun je downloaden op deze site: https://www.jetbrains.com/pycharm/download/other.html.
3: Open Anaconda navigator en kijk of je in het overzicht nu PyCharm community kunt vinden, anders moet je misschien even opnieuw opstarten. Letop! PyCharm proffesional
   staat er waarschijnlijk ook tussen, die moet je niet hebben.
4: Open PyCharm community via Anaconda navigator
5: Als het goed is zie je nu een menu met "Welcome to PyCharm". Kies "Get from VCS".
6: Je ziet nu twee velden waar je iets in kunt vullen, plak deze link: https://github.com/Lstarink/vector_tomography.git bij URL.
7: Onder die twee velden zie je een melding staan dat je git niet hebt geinstalleerd, klik op install git.
8: Klik nu rechtsonder op Clone, en daarna Trust Project.
9: Als het goed is kun je nu bij al mijn code. Open het mapje vector_tomography aan de linker kant bij project. Open het mapje source, en open main.py.
10: Klik helemaal rechts onder op <No interpreter>. Daarna op add interpreter. 
    Als het goed is opent nu een nieuw window en staat er bij Base interpreter het path waar je python hebt geinstalleerd
    Selecteer New enviroment.
    Klik op OK.
11: Klik rechts bovenin op Add Configuration..., klik op Add new run configuration, selecteer python. 
    Ze bij Name: "main".
    Bij Script path moet je rechts op het mapjeklikken, ga naar >source, klik op main.py en klik op OK. 
    Bij interpreter verifieer dat er Project Default (...) staat, selecteer anders een python interpeter.
    Klik op OK.
12: Ga links onder naar Python Packages. Zoek en installeer de volgende packages
    Numpy
    Sympy
    Matplotlib
13: Ga onderin naar Terminal en type:
    git checkout erol
    of:
    git checkout willem
13: Nu kun je rechts boven op het groene driehoekje klikken om het model te runnen
  

Als alles gelukt is kunnen jullie nu het model runnen. Ik heb het model zo geschreven dat jullie enkel en alleen dingen aan hoeven te passen in de settings.py file. 
Je hebt niet alle settings nodig, hier een overzicht van wat je in kunt stellen, wat het doet en waar je van af moet blijven.

-Settings for what needs to be run- 

  Het model bestaat uit 3 delen. Je kunt kiezen welke je runt bovenaan bij """Settings for what needs to be run""".

  Deel 1 berekent alle eigenschappen van de setup. Het maakt de gram matrix en rekent uit waar alle kruispunten liggen. Dit deel doet niets met metingen.
    Afhankelijk van de instellingen voor de Gram matrix kan dit erg lang duren om te runnen. Daarom heb ik het zo gemaakt dat de gram matrix van een setup
    opgeslagen wordt als je dit deel een keer gerunt hebt. Zet dus altijd de eerste keer dat je een nieuwe setup gebruikt of instellingen voor de gram matrix of kruispunten
    aanpast "recalulate_gram_matrix = True". Daarna kan  "recalulate_gram_matrix = False" om tijd te besparen. Als je alléén dit deel van de code wilt runnen omdat je
    bijvoorbeeld een nieuwe gram matrix wilt maken, maar niet perse iets wilt doen met metingen, dan kun je "only_calculate_setup" op "True" zetten.

  Deel 2 maakt metingen. Het neemt een vector veld wat je zelf in kunt vullen en "meet" dit met de setup die je hebt opgegeven.
    Met simuleren gebruiken we alleen door het model gegenereerde metingen. Zet daarom "generate_your_own_measurement" altijd op "True". 
    Als je de metingen wilt runnen die we in het lab gemaakt hebben om er zelf naar te kijken, zet deze dan op "False". Verderop staat wat je nog meer aan moet passen.

  Deel 3 Rekent alles uit, maakt de plotjes van de vector velden en de error. Het maakt alle resultaten.
    Dit deel runt in principe altijd. Tenzij je "only_calculate_setup" op "True" hebt gezet.
 
 -Settings for what is used en where to save it-
  
  In dit deel van de settings bepaal je welke setup je gebruikt en waar je het op slaat.
  
  Bij FileName zet je de naam van het csv bestand met de lijnen die de setup maken.
  Ik heb al een aantel setups erin staan die je kunt gebruiken of mee kunt proberen, zoals bijv:
  - final_setup.csv; de setup waarmee we gemeten hebben in de windtunnel.
  - final_setup_alleen_recht.csv; De naam zegt het.
  - 3D_setup2.csv; De "ideale setup", een 5x5 grid van lijnen in x y en z richting
  - 3D_setup4.csv; De setup met twee vlakken die we in het begin van de BEP hadden bedacht.
  Als je zelf een setup maakt, zorg ervoor dat:
  - Ze dezelfde opzet hebben als de andere setups, een csv bestand met per rij lijnnr, Ax,Ay,Az,Bx,By,Bz
  - Ze opgeslagen zijn als csv bestand
  - Je ze in het mapje "Setups" zet.
  
  Bij Name_of_calculations kun je zetten waar alles opgeslagen wordt. Het model maakt een mapje mat deze naam in het mapje output. Hier wordt dan je plotjes, gram matrix
  en kruispunten opgeslagen. Het is verstandig om elke keer dat je aanpassigen doet aan de instellingen dit een andere naam te geven, en de naam zo te geven dat 
  het duidelijk is welke instellingen je hebt gebruikt. Als je dit niet verandert en je runt het model met nieuwe instellingen worden je oude resultaten vervangen door nieuwe 
  en ben je ze kwijt!
  
  Bij het else statement van measurement file kun je eventueel onze échte metingen gebruiken. Zet dan hier de naam van het bestand dat je wilt gebruiken.
  Ik heb in het mapje Measurements onze metingen gezet. Zet dan natuurlijk "generate_your_own_measurements" op "False"!
  
  BESCHRIJF WAT WELKE METING IS!!
 
-Define your own vector field-
 Zet hier bij u, v, w het vector veld dat je wilt gebruiken als  "generate_your_own_measurements" op "True" staat. Alles is in meters per seconde!
 
-Settings for intersections-
  Met de "intersection_boundary_edge" opties kun je kruispunten aan de randen eraf knippen. Dit heb ik gemaakt zodat het model de sensors zelf niet als kruispunt
  ziet. Zet deze instelling zo dat het model alle sensors "eraf knipt". Of dit goed gaat kun je checken door in de plot instellingen "plot_line_intersections" op
  "True" te zetten. Voorbeeld: de setup waarmee we in de windtunnel gemeten hebben heeft sensors op de randen van x en y, maar niet op z. Zet dus z op 0 en x en y 
  op een kleine waarde, bijv 0.005m.
  
  Met "use_only_full_rank_intersections" kun je ervoor kiezen om alleen kruispunten te gebruiken van volle rang. Dit zijn kruispunten waarvan de tubes die er 
  door heen gaan R3 op spannen, het is interessant om te kijken of deze instelling de resultaten verbetert!

-Settings for error of the sensors
  Met deze settings kun je de sensor error modelleren.
  Als je sensors zonder error wilt zet je "use_sensor_error" op "false".
  Anders modelleert het model de sensors met de standaarddeviatie die je opgeeft bij "sensor_stddev". Het is nuttig om uit te zoeken tot welke standaard deviatie
  het model blijft functioneren, eg. om te kijken wanneer de error te groot wordt.
  
-Settings for Gram matrix
  use_integration_for_gram_matrix: In principe wil je deze altijd op True hebben staan tenzij je heel snel iets anders uit wilt proberen,
  anders gebruikt het model een vereenvoudigde methode.
  
  matrix_integration_setting: Als je settings aan het uitproberen bent kun je dit op 20 houden. Als je een resultaat wilt maken dat we in ons paper gaan zetten,
  zet dit dan op 50. Deze setting bepaalt hoe nauwkeurig de gram matrix uitgerekend wordt en bepaalt hoe lang het duurt om het model te runnen. Op 20 duurt het
  ongeveer 1.5 minuut om het te runnen, op 50 ongeveer een kwartier, afhankelijk van het aantal lijnen in je setup. Als je echt een grote setup uit wilt rekenen
  kun je mij appen, dan reken ik de gram matrix uit op de server en stuur ik hem naar je terug.
  
  tube_width: Waarschijnlijk de belangrijkste instelling! Hier stel je de diameter van de tube in. In de python console wordt info geprint over de kruispunten,
  kijk hier goed naar. Je wilt de tube_width zo instellen dat je zo veel mogelijk full rank intersections hebt, maar zo min mogelijk intersections met 4, 5 of meer
  tubes!

 -Settings for interpolation
  de interpolation_offset wordt gebruikt om een buitenrand te definieren waar niet geinterpoleerd wordt. Het model is erg onnauwkeurig op de buitenrand van de setup
  Hier ben je namelijk eigenlijk aan het extrapoleren in plaats van interpoleren omdat je je buiten de kruispunten bevindt. Stel dit zo in dat je de buitenrand eraf
  knipt. 
  Let op! Het model rekent de grootte van de setup uit en definieert het als een rechthoekig volume van x_min tot x_max, y_min tot y_max en z_min tot z_max.
  Deze waarden worden bepaald aan de hand waar de sensors staan. Dus x_min krijgt de waarde van de sensor met het kleinste x coordinaat en x_max krijgt de waarde 
  van de sensor met het grootste x coordinaat.
  Voorbeeld: bij de setup die we gemaakt hebben om in de windtunnel te testen ligt het eerste kruispunt op (0.033, 0.033, 0.033). De kleinste voorkomende x waarde 
  van een sensor is 0.0, de kleinste voorkomende y waarde is 0.0 en de kleinste voorkomende z waarde is 0.033
  zet dus interpolation_offset_x = 0.03 en interpolation_offset_y = 0.03 (ietsje kleiner dan 0.033 om dat kruispunt wel mee te nemen)
  zet interpolation_offset_z = 0.0
  
 -Settings for plotting
  Hier zet je aan welke plotjes je wilt zien of opslaan. jullie moeten ze een voor een maar aanzetten om te kijken wat ze precies doen, sommigen zijn nuttig anderen niet.
  
  plot_field_sliced, plot_error_sliced en calculate_error zijn waarschijnlijk het meest interessant.
  Als je save_figures op True hebt staan worden deze plotjes opgeslagen in >output>calculations+ de naam die je zelf hebt opgegeven bij Name_of_calculations
  calculate_error maakt geen plotje, maar print in de python console de global error en intersection error.
 
 Samengevat:
   Kijk met behulp van tube_width, use_only_full_rank_intersections en sensor_stddev wat de mogelijkheden zijn voor een setup. En verzamel data in de vorm van 
   plotjes en waardes voor global error en intersection error.
  
   Interessante vragen over een setup lijken mij: 
    - Tot welke sensor_stddev functioneert een setup; Tot welke sensor_stddev blijft de error< x%
    - Wat is een goede verhouding voor aantal diagonale lijnen in een setup tov rechte lijnen?
    - Welke waarden voor error zijn haalbaar met dit model?
    - Wat is de optimale tube_width voor een model?
    - Wat is de invloed van use_only_full_rank_intersections op de prestatie van een setup?
    - Voor wat voor vector velden werkt dit model goed of niet goed?
    - Waaraan moet een goede setup voldoen?
   
   LETOP! dit kan fout gaan
    - Je hebt voor een setup een gram matrix uitgerekend met "recalculate_gram_matrix = True"
      Je wilt daarna rustig klooien met plotjes en zet daarom "recalculate_gram_matrix = False"
      Je doet aanpassingen aan tube_width of je setup maar vergeet de gram matrix opnieuw uit te rekenenen met "recalculate_gram_matrix = True"
      Het model gebruikt nu een oude gram matrix met instellingen die niet meer van toepassing zijn en gaat fouten maken!
    
    Als je twijfelt aan of het model nog wel werkt:
      - Selecteer bij FileName 3D_setup2.csv
      - Geef Name_of_calculations een nieuwe naam
      - zet "recalculate_gram_matrix = True"
      - Neem het vectorveld x = sp.symbols('x')
                            y = sp.symbols('y')
                            z = sp.symbols('z')
      Het model moet nu bij intersection_error een erg klein getal aangeven, ordegrote 10^-13                                                                                       
                                                                                                 

                                                                                               
   
    
                                                                                               
   
                                                                                                
               
                                                                                                 
                                                                                             
  
  
 


  
  




