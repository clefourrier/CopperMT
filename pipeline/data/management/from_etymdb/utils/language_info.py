#!/usr/bin/env python
"""For each language, contains its ancestor and abbreviation to name mapping"""

#  Full code to name list:
#  https://en.wiktionary.org/wiki/Wiktionary:List_of_languages
name = {
    "pl": "Polish",
    "cs": "Czech",
    "lt": "Lithuanian",
    "la": "Latin",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "ro": "Romanian",
}

available_espeak_languages = {"pl": "pl",
                              "cs": "cs",
                              "la": "la",
                              "it": "it",
                              "es": "es",
                              "fr": "fr",
                              "ro": "ro",
                              "pt": "pt-pt",
                              "gl": "pt-pt",
                              "ca": "ca",
                              "oc": "ca",
                              "rup": "ro",
                              "dlm": "ro",
                              "itc-ola": "la"}

ancestors = {
    # ---- SLAVIC
    # PIE, PBS, PSlavic, (PWestSlavic), OPol, (MPol).
    "pl": ['ine-pro', 'ine-bsl-pro', 'sla-pro', 'zlw-opl'],
    # PIE, PBS, PSlavic, (PWestSlavic), OCS, (MCS).
    "cs": ['ine-pro', 'ine-bsl-pro', 'sla-pro', 'zlw-ocs'],
    # PIE, PBS, (Proto-baltic, Proto-east-baltic), OLith, (MLith).
    "lt": ['ine-pro', 'ine-bsl-pro', 'olt'],
    # ---- ROMANCE
    # PIE, (PItalo-Celtique), Proto-Italique, (CommonLatinFalsique)
    # , OLatin, latin/vulg-latin, (old ita, mid ita).
    "it": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul'],
    "dlm": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul'],
    "la": ['ine-pro', 'itc-pro', 'itc-ola'],
    "es": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'osp'],
    "ca": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'pro', 'roa-oca'],
    "oc": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'pro'],
    "fr": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr', 'frm'],
    "frm": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr'],
    "ofr": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul'],
    "ro": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul'],
    "rup": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul'],
    "pt": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'roa-opt'],
    "gl": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'roa-opt'],
    "nrf": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr', 'frm'],
    # ---- GERMANIC
    "en": ['ine-pro', 'gem-pro', 'gmw-pro', 'ang', 'enm'],
    # ---- INDO-ARIAN
    "hi": ['ine-pro', 'iir-pro', 'inc-pro', 'sa', 'inc-ash', 'psu', 'inc-sap', 'inc-ohi'],
    "sa": ['ine-pro', 'iir-pro', 'inc-pro'],
    "bn": ['ine-pro', 'iir-pro', 'inc-pro', 'sa', 'inc-ash', 'inc-mgd', 'inc-obn', 'inc-mbn'],
    # ---- FRENCH CREOLES
    "gcf": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr', 'frm', 'fr'],
    "gcr": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr', 'frm', 'fr'],
    "lou": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr', 'frm', 'fr'],
    "rcf": ['ine-pro', 'itc-pro', 'itc-ola', 'la', 'lat-vul', 'ofr', 'frm', 'fr'],

}