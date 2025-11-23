<?php
/**
 * Plugin Name: Rayvia Video Translator
 * Description: Embed Rayvia AI Video Translator/Converter tool inside WordPress.
 * Version: 1.0
 * Author: Rayvia
 */

if ( ! defined( 'ABSPATH' ) ) exit; // Exit if accessed directly

// Shortcode: [rayvia_video_translator]
function rayvia_video_translator_shortcode( $atts ) {
    $atts = shortcode_atts( array(
        // Yahan apne Python / AI tool ka URL daalna hai jahan app chal raha ho
        'src'    => 'https://example.com', // FIXME: baad me change karna
        'height' => '900px',
    ), $atts, 'rayvia_video_translator' );

    ob_start();
    ?>
    <div style="width:100%; max-width:1280px; margin:20px auto; border:1px solid #ddd; border-radius:12px; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,0.08);">
        <div style="padding:14px 20px; background:#111827; color:#f9fafb; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:20px; font-weight:700;">Rayvia Video Translator & Converter</div>
                <div style="font-size:12px; opacity:0.8;">Translate & dub online videos in any language – Full HD output*</div>
            </div>
            <div style="font-size:11px; text-align:right; opacity:0.8;">
                Powered by Rayvia AI Engine<br>
                <span style="font-size:10px;">Use only with videos you own the rights to.</span>
            </div>
        </div>
        <iframe 
            src="<?php echo esc_url( $atts['src'] ); ?>" 
            style="width:100%; border:0;" 
            height="<?php echo esc_attr( $atts['height'] ); ?>" 
            loading="lazy"
            allowfullscreen>
        </iframe>
    </div>
    <?php
    return ob_get_clean();
}
add_shortcode( 'rayvia_video_translator', 'rayvia_video_translator_shortcode' );
