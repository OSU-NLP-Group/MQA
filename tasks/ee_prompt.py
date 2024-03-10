ee_label_prompt = 'Extract the key words from the sentence that prompt action/activity type "{}".'

label2demo = {}

output_format = '''
Sentence: "{}"
Output => 
'''

label2demo['Conflict:Demonstrate'] = '''
Sentence: "Demonstrators rally against Venezuela's President Nicolas Maduro in Caracas, Venezuela, April 13, 2017"
Output => rally
Sentence: "FILE - Police arrest Black Lives Matter leader Deray McKesson during a protest along Airline Highway, a major road that passes in front of the Baton Rouge Police Department headquarters, July 9, 2016"
Output => protest
Sentence: "Sister marches were held in more than 600 locations Saturday in the U.S. and across the globe in solidarity with the marchers in Washington"
Output => marches
'''

label2demo["Justice:Arrest-Jail"] = '''
Sentence: "Trump's First Big Immigration Op: 680 Arrested and Lots of Questions-The Immigration and Customs Enforcement Agency (ICE) says it rounded up 680 undocumented immigrants last week in enforcement actions in multiple locations around the U.S"
Output => arrested, rounded
Sentence: "Rubio, warning the U.S. faces an unprecedented threat from the Islamic State group, said he would go after terrorists \"wherever they are\" and if caught, \"we're sending them to Guantanamo,\" referring to the detention center for prisoners in Cuba"
Output => caught
Sentence: "Despite minimal coverage in the dominant state media, tens of thousands took part and hundreds were detained in Moscow, including 20-year-old political science major David-Viktor Ratkin"
Output => detained
'''

label2demo["Movement:Transport"] = '''
Sentence: "FILE - Afghan National Army troops arrive in Nad Ali district of Helmand province, southern Afghanistan, Aug. 10, 2016"
Output => arrive
Sentence: "Trucks carry tents and construction material to be used for make-shift refugee housing, in Oncupinar, Turkey, Feb. 8, 2016"
Output => carry
Sentence: "According to the United Nations, more than 43,000 migrants, mostly from sub-Saharan Africa, have reached European shores this year from Libya and a surge in migrant crossings is expected this spring and summer"
Output => reach
'''


label2demo["Life:Die"] = '''
Sentence: "The U.S. does not have information on who was behind the attack that killed Badreddine, State Department spokesman John Kirby said"
Output => killed
Sentence: "A Second Night of Chicago Protests Over Police Shooting-Protesters took to the streets of Chicago for the second straight night Wednesday, one day after the city's police department released video of a black teenager being shot to death by a white officer",
Output => death
Sentence: "Police survey the crime scene where Kem Ley was shot dead while family members pray next to his dead body, July 10, 2016"
Output => dead
'''

label2demo["Contact:Phone-Write"] = '''
Sentence: "In a statement released by the Cruz campaign, Bush called Cruz the only hope for Republicans to win back the White House and criticized Trump"
Output => called
Sentence: "Democratic U.S. presidential candidate Hillary Clinton is accompanied by her daughter Chelsea Clinton (R) and her husband, former U.S. President Bill Clinton, as she speaks to supporters at her final 2016 New Hampshire presidential primary night rally"
Output => speaks
Sentence: "Trump-Putin Phone Call: 'A Toe in the Water'-MOSCOW \u2014\u00a0-Saturday's phone conversation between presidents Donald Trump and Vladimir Putin has raised the hopes of many Russian politicians for a U.S.-Russian rapprochement"
Output => conversation
'''

label2demo["Conflict:Attack"] = '''
Sentence: "Charlotte Police Chief Kerr Putney speaks as city officials, including Charlotte mayor Jennifer Roberts (R), listen during a news conference following the fatal police shooting of Keith Lamont Scott, in Charlotte, North Carolina, Sept. 22, 2016"
Output => shooting
Sentence: "\"The fact that we will necessarily cooperate in the fight against ISIS helps us find some common ground"
Output => fight
Sentence:"A destroyed van is pictured near a Turkish police bus which was targeted in a bomb attack in a central Istanbul district, Turkey, June 7, 2016"
Output => attack
'''

label2demo["Transaction:Transfer-Money"] = '''
Sentence: "Moscow turns up pressure on Kyiv-Russian state-controlled gas company Gazprom has reiterated its threat to stop supplying Ukraine with gas if it does not pay in advance for June deliveries, Russian news agencies reported on Monday"
Output => pay
Sentence: "Obama requested $3.45 billion in the 2017 budget to help fund Afghan security forces"
Output => fund
Sentence:  "As a candidate, Trump has said he\u2019d step up the bombing campaign to \u201cknock out\u201d IS and stop buying oil from Saudi Arabia if it doesn\u2019t contribute troops to the fight against the Islamic State"
Output => buying
'''

label2demo["Contact:Meet"] = '''
Sentence: "North Korea against world-U.S. President Barack Obama, South Korean President Park Geun-hye and Japanese Prime Minister Shinzo Abe met at the Nuclear Security Summit in Washington on Thursday to underscore their united commitment to exert increasing pressure on North Korea to abandon its nuclear program"
Output => met
Sentence: "Richard Holbrooke, the U.S. special representative to Pakistan and Afghanistan, listens to a question during a news conference in Islamabad June 19, 2010"
Output => conference
Sentence: "Rev. Jimmy Jones, founder of People's Temple, clasps an unidentified man at Jonestown, Nov. 18, 1978, during Congressman' Leo J. Ryan's visit"
Output => visit
'''

